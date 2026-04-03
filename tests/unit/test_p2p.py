# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for nkipy.p2p.endpoint and nkipy.p2p.transfer."""

import threading
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Fixtures: mock uccl.p2p before importing nkipy.p2p ──────────


class FakeDesc:
    """Minimal stand-in for a UCCL transfer descriptor."""

    def __init__(self, mr_id):
        self.mr_id = mr_id


class FakeEndpoint:
    """Minimal stand-in for uccl.p2p.Endpoint."""

    def __init__(self, nc_idx):
        self.nc_idx = nc_idx
        self._next_mr = 0
        self.dereg_calls = []

    def register_memory(self, handles):
        descs = []
        for _ in handles:
            descs.append(FakeDesc(self._next_mr))
            self._next_mr += 1
        return descs

    def dereg(self, mr_id):
        self.dereg_calls.append(mr_id)

    def get_metadata(self):
        return b"\xab\xcd"

    def get_serialized_descs(self, descs):
        return b"\x01\x02"

    def start_passive_accept(self):
        pass

    def add_remote_endpoint(self, metadata):
        return True, 42

    def deserialize_descs(self, data):
        # Return same number of descs as were registered
        return [FakeDesc(i) for i in range(self._next_mr)]

    def transfer(self, conn_id, mode, local, remote):
        return True, None


@pytest.fixture(autouse=True)
def _mock_uccl(monkeypatch):
    """Inject a fake uccl.p2p module so we never touch real RDMA."""
    import sys

    fake_p2p = types.ModuleType("uccl.p2p")
    fake_p2p.Endpoint = FakeEndpoint
    fake_uccl = types.ModuleType("uccl")
    fake_uccl.p2p = fake_p2p

    monkeypatch.setitem(sys.modules, "uccl", fake_uccl)
    monkeypatch.setitem(sys.modules, "uccl.p2p", fake_p2p)

    # Force reimport so the module picks up the fake
    for mod_name in list(sys.modules):
        if mod_name.startswith("nkipy.p2p"):
            del sys.modules[mod_name]

    yield

    for mod_name in list(sys.modules):
        if mod_name.startswith("nkipy.p2p"):
            del sys.modules[mod_name]


SAMPLE_BUFFERS = [
    ("weight_a", 0x1000, 256),
    ("weight_b", 0x2000, 512),
]


# ═══════════════════════════════════════════════════════════════════
# endpoint._VAHandle
# ═══════════════════════════════════════════════════════════════════


class TestVAHandle:
    def test_data_ptr(self):
        from nkipy.p2p.endpoint import _VAHandle

        h = _VAHandle(0xDEAD, 1024)
        assert h.data_ptr() == 0xDEAD

    def test_numel_and_element_size(self):
        from nkipy.p2p.endpoint import _VAHandle

        h = _VAHandle(0, 4096)
        assert h.numel() == 4096
        assert h.element_size() == 1


# ═══════════════════════════════════════════════════════════════════
# endpoint._get_nc_idx
# ═══════════════════════════════════════════════════════════════════


class TestGetNcIdx:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
        monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "0")
        from nkipy.p2p.endpoint import _get_nc_idx

        assert _get_nc_idx() == 0

    def test_multi_core(self, monkeypatch):
        monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "3,4,5")
        from nkipy.p2p.endpoint import _get_nc_idx

        assert _get_nc_idx() == 3


# ═══════════════════════════════════════════════════════════════════
# endpoint.RankEndpoint
# ═══════════════════════════════════════════════════════════════════


class TestRankEndpoint:
    def _make_ep(self):
        from nkipy.p2p.endpoint import RankEndpoint

        return RankEndpoint(nc_idx=0)

    # -- register --------------------------------------------------

    def test_register_populates_state(self):
        ep = self._make_ep()
        ep.register(SAMPLE_BUFFERS)

        assert ep.registered
        assert len(ep.xfer_descs) == 2
        assert ep.buf_info == [("weight_a", 256), ("weight_b", 512)]
        assert ep.ep is not None

    def test_register_is_idempotent(self):
        ep = self._make_ep()
        ep.register(SAMPLE_BUFFERS)
        first_descs = ep.xfer_descs

        ep.register([("other", 0x9999, 64)])  # should be no-op
        assert ep.xfer_descs is first_descs

    def test_register_creates_endpoint_lazily(self):
        ep = self._make_ep()
        assert ep.ep is None
        ep.register(SAMPLE_BUFFERS)
        assert isinstance(ep.ep, FakeEndpoint)

    # -- registered property ---------------------------------------

    def test_registered_false_initially(self):
        ep = self._make_ep()
        assert not ep.registered

    # -- dereg_sync ------------------------------------------------

    def test_dereg_sync_clears_state(self):
        ep = self._make_ep()
        ep.register(SAMPLE_BUFFERS)
        fake_ep = ep.ep

        ep.dereg_sync()

        assert ep.ep is None
        assert ep.xfer_descs == []
        assert ep.buf_info == []
        assert not ep.registered
        assert fake_ep.dereg_calls == [0, 1]

    def test_dereg_sync_noop_when_empty(self):
        ep = self._make_ep()
        ep.dereg_sync()  # should not raise
        assert ep.ep is None

    def test_dereg_sync_waits_for_pending_async(self):
        ep = self._make_ep()
        ep.register(SAMPLE_BUFFERS)

        barrier = threading.Event()
        original_bg = None

        # Patch dereg_async to use a slow background thread
        ep.dereg_async()
        # Immediately re-register and dereg_sync — sync must wait
        ep.register([("c", 0x3000, 128)])
        ep.dereg_sync()
        assert not ep.registered

    # -- dereg_async -----------------------------------------------

    def test_dereg_async_clears_main_thread_state_immediately(self):
        ep = self._make_ep()
        ep.register(SAMPLE_BUFFERS)

        ep.dereg_async()

        # Main-thread state cleared right away
        assert ep.ep is None
        assert ep.xfer_descs == []
        assert ep.buf_info == []

    def test_dereg_async_deregisters_in_background(self):
        ep = self._make_ep()
        ep.register(SAMPLE_BUFFERS)
        fake_ep = ep.ep

        ep.dereg_async()
        ep.wait()

        assert fake_ep.dereg_calls == [0, 1]

    def test_dereg_async_noop_when_no_endpoint(self):
        ep = self._make_ep()
        ep.dereg_async()  # should not raise
        ep.wait()

    # -- wait ------------------------------------------------------

    def test_wait_returns_immediately_when_no_thread(self):
        ep = self._make_ep()
        ep.wait()  # should not block

    def test_wait_joins_background_thread(self):
        ep = self._make_ep()
        ep.register(SAMPLE_BUFFERS)
        ep.dereg_async()
        ep.wait()
        assert ep._dereg_thread is None

    # -- full lifecycle: register → dereg_async → register ---------

    def test_reregister_after_async_dereg(self):
        ep = self._make_ep()
        ep.register(SAMPLE_BUFFERS)
        ep.dereg_async()

        # Second register should wait for async, then create fresh state
        new_bufs = [("w", 0x5000, 64)]
        ep.register(new_bufs)

        assert ep.registered
        assert ep.buf_info == [("w", 64)]
        assert len(ep.xfer_descs) == 1

    def test_multiple_async_dereg_cycles(self):
        ep = self._make_ep()
        for i in range(3):
            ep.register([("w", 0x1000 * i, 128)])
            ep.dereg_async()
        ep.wait()
        assert not ep.registered


# ═══════════════════════════════════════════════════════════════════
# transfer.collect_weight_buffers
# ═══════════════════════════════════════════════════════════════════


class TestCollectWeightBuffers:
    def test_delegates_to_model(self):
        from nkipy.p2p.transfer import collect_weight_buffers

        model = MagicMock()
        model.weight_buffers.return_value = iter(SAMPLE_BUFFERS)

        result = collect_weight_buffers(model)

        model.weight_buffers.assert_called_once()
        assert result == list(SAMPLE_BUFFERS)


# ═══════════════════════════════════════════════════════════════════
# transfer.WeightServer
# ═══════════════════════════════════════════════════════════════════


class TestWeightServer:
    def _make_model(self):
        model = MagicMock()
        model.weight_buffers.return_value = iter(SAMPLE_BUFFERS)
        return model

    def test_init_registers_buffers(self):
        from nkipy.p2p.transfer import WeightServer, rank_endpoint

        rank_endpoint.dereg_sync()  # clean slate
        ws = WeightServer(self._make_model())

        assert rank_endpoint.registered
        assert len(rank_endpoint.xfer_descs) == 2

    def test_get_weight_info(self):
        from nkipy.p2p.transfer import WeightServer, rank_endpoint

        rank_endpoint.dereg_sync()
        ws = WeightServer(self._make_model())
        info = ws.get_weight_info()

        assert "weights" in info
        assert "metadata" in info
        assert len(info["weights"]) == 2
        assert isinstance(info["metadata"], str)

    def test_cleanup_waits(self):
        from nkipy.p2p.transfer import WeightServer, rank_endpoint

        rank_endpoint.dereg_sync()
        ws = WeightServer(self._make_model())
        ws.cleanup()  # should not raise


# ═══════════════════════════════════════════════════════════════════
# transfer.push_to_peer
# ═══════════════════════════════════════════════════════════════════


class TestPushToPeer:
    @patch("nkipy.p2p.transfer.dist")
    def test_push_registers_transfers_and_deregs(self, mock_dist):
        from nkipy.p2p.endpoint import RankEndpoint
        from nkipy.p2p.transfer import push_to_peer

        mock_dist.get_rank.return_value = 0
        mock_dist.broadcast_object_list.side_effect = (
            lambda obj, src: None
        )

        per_rank_info = [("abcd", "0102")]
        ep = RankEndpoint(nc_idx=0)

        # broadcast_object_list is a no-op, so pre-fill obj_list
        # by patching to set the value
        def fake_broadcast(obj_list, src):
            obj_list[0] = per_rank_info

        mock_dist.broadcast_object_list.side_effect = fake_broadcast

        push_to_peer(ep, SAMPLE_BUFFERS, per_rank_info)

        # After push, endpoint should be deregistered
        assert not ep.registered
        assert ep.ep is None
        mock_dist.barrier.assert_called_once()

    @patch("nkipy.p2p.transfer.dist")
    def test_push_asserts_on_connection_failure(self, mock_dist):
        from nkipy.p2p.endpoint import RankEndpoint
        from nkipy.p2p.transfer import push_to_peer

        mock_dist.get_rank.return_value = 0

        def fake_broadcast(obj_list, src):
            obj_list[0] = [("abcd", "0102")]

        mock_dist.broadcast_object_list.side_effect = fake_broadcast

        ep = RankEndpoint(nc_idx=0)

        # Make add_remote_endpoint fail
        ep.register(SAMPLE_BUFFERS)
        ep.ep.add_remote_endpoint = lambda m: (False, None)

        with pytest.raises(AssertionError, match="Failed to connect"):
            push_to_peer(ep, SAMPLE_BUFFERS, [("abcd", "0102")])


# ═══════════════════════════════════════════════════════════════════
# transfer.receive_from_peer
# ═══════════════════════════════════════════════════════════════════


class TestReceiveFromPeer:
    @patch("nkipy.p2p.transfer.requests")
    @patch("nkipy.p2p.transfer.dist")
    def test_receive_registers_and_posts(self, mock_dist, mock_requests):
        from nkipy.p2p.endpoint import RankEndpoint
        from nkipy.p2p.transfer import receive_from_peer

        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 1
        mock_dist.gather_object.side_effect = (
            lambda obj, gathered, dst: gathered.__setitem__(
                0, (obj.hex() if isinstance(obj, bytes) else obj)
            )
            if gathered is not None
            else None
        )

        ep = RankEndpoint(nc_idx=0)
        receive_from_peer(ep, SAMPLE_BUFFERS, "http://peer:8080")

        assert ep.registered
        mock_requests.post.assert_called_once()
        call_url = mock_requests.post.call_args[0][0]
        assert call_url == "http://peer:8080/p2p_push_weights"
        mock_dist.barrier.assert_called_once()

    @patch("nkipy.p2p.transfer.requests")
    @patch("nkipy.p2p.transfer.dist")
    def test_receive_custom_endpoint(self, mock_dist, mock_requests):
        from nkipy.p2p.endpoint import RankEndpoint
        from nkipy.p2p.transfer import receive_from_peer

        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 1
        mock_dist.gather_object.side_effect = (
            lambda obj, gathered, dst: gathered.__setitem__(0, obj)
            if gathered is not None
            else None
        )

        ep = RankEndpoint(nc_idx=0)
        receive_from_peer(
            ep, SAMPLE_BUFFERS, "http://peer:8080/", "/custom_push"
        )

        call_url = mock_requests.post.call_args[0][0]
        assert call_url == "http://peer:8080/custom_push"


# ═══════════════════════════════════════════════════════════════════
# transfer.fetch_tok_embedding
# ═══════════════════════════════════════════════════════════════════


class TestFetchTokEmbedding:
    @patch("nkipy.p2p.transfer.dist")
    @patch("nkipy.p2p.transfer.requests")
    def test_rank0_fetches_and_broadcasts(self, mock_requests, mock_dist):
        from nkipy.p2p.transfer import fetch_tok_embedding

        mock_dist.get_rank.return_value = 0

        # Simulate HTTP response with a float32 tensor
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        resp = MagicMock()
        resp.headers = {
            "X-Shape": "2,2",
            "X-Dtype": "float32",
        }
        resp.content = data.tobytes()
        mock_requests.get.return_value = resp

        captured = {}

        def fake_broadcast(obj_list, src):
            captured["tensor"] = obj_list[0]

        mock_dist.broadcast_object_list.side_effect = fake_broadcast

        fetch_tok_embedding("http://peer:8080")

        mock_requests.get.assert_called_once_with(
            "http://peer:8080/tok_embedding"
        )
        assert captured["tensor"].shape == (2, 2)
        np.testing.assert_allclose(
            captured["tensor"].numpy(), data, rtol=1e-5
        )
