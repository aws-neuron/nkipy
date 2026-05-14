# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the NKIPy vLLM plugin components."""

import os
import unittest
from unittest.mock import MagicMock, patch


class TestPluginRegister(unittest.TestCase):
    """Test the plugin registration entry point."""

    @patch("nkipy.vllm_plugin._is_neuron_dev", return_value=False)
    def test_register_no_device(self, mock_dev):
        from nkipy.vllm_plugin import register

        result = register()
        assert result is None

    @patch("nkipy.vllm_plugin._is_neuron_dev", return_value=True)
    def test_register_neuron_device(self, mock_dev):
        from nkipy.vllm_plugin import register

        result = register()
        assert result == "nkipy.vllm_plugin.platform.NKIPyPlatform"


class TestNKIPyPlatform(unittest.TestCase):
    """Test the NKIPy platform class."""

    def test_platform_enum(self):
        from nkipy.vllm_plugin.platform import NKIPyPlatform
        from vllm.platforms import PlatformEnum

        assert NKIPyPlatform._enum == PlatformEnum.OOT

    def test_device_type(self):
        from nkipy.vllm_plugin.platform import NKIPyPlatform

        assert NKIPyPlatform.device_type == "neuron"

    def test_get_device_name(self):
        from nkipy.vllm_plugin.platform import NKIPyPlatform

        assert NKIPyPlatform.get_device_name(0) == "neuron:0"
        assert NKIPyPlatform.get_device_name(3) == "neuron:3"

    def test_is_pin_memory_available(self):
        from nkipy.vllm_plugin.platform import NKIPyPlatform

        assert NKIPyPlatform.is_pin_memory_available() is False

    def test_check_and_update_config_sets_worker(self):
        from nkipy.vllm_plugin.platform import NKIPyPlatform

        vllm_config = MagicMock()
        vllm_config.parallel_config.worker_cls = "auto"
        vllm_config.parallel_config.distributed_executor_backend = "mp"
        vllm_config.cache_config.block_size = None
        vllm_config.scheduler_config.async_scheduling = False

        NKIPyPlatform.check_and_update_config(vllm_config)

        assert vllm_config.parallel_config.worker_cls == "nkipy.vllm_plugin.worker.NKIPyWorker"
        assert vllm_config.cache_config.block_size == 16

    def test_check_and_update_config_preserves_custom_worker(self):
        from nkipy.vllm_plugin.platform import NKIPyPlatform

        vllm_config = MagicMock()
        vllm_config.parallel_config.worker_cls = "my.custom.Worker"
        vllm_config.parallel_config.distributed_executor_backend = "mp"
        vllm_config.cache_config.block_size = 32
        vllm_config.scheduler_config.async_scheduling = False

        NKIPyPlatform.check_and_update_config(vllm_config)

        assert vllm_config.parallel_config.worker_cls == "my.custom.Worker"
        assert vllm_config.cache_config.block_size == 32

    def test_check_and_update_config_fixes_executor_backend(self):
        from nkipy.vllm_plugin.platform import NKIPyPlatform

        vllm_config = MagicMock()
        vllm_config.parallel_config.worker_cls = "auto"
        vllm_config.parallel_config.distributed_executor_backend = "ray"
        vllm_config.cache_config.block_size = None
        vllm_config.scheduler_config.async_scheduling = False

        NKIPyPlatform.check_and_update_config(vllm_config)

        assert vllm_config.parallel_config.distributed_executor_backend == "mp"

    def test_check_and_update_config_disables_async(self):
        from nkipy.vllm_plugin.platform import NKIPyPlatform

        vllm_config = MagicMock()
        vllm_config.parallel_config.worker_cls = "auto"
        vllm_config.parallel_config.distributed_executor_backend = "mp"
        vllm_config.cache_config.block_size = None
        vllm_config.scheduler_config.async_scheduling = True

        NKIPyPlatform.check_and_update_config(vllm_config)

        assert vllm_config.scheduler_config.async_scheduling is False


class TestNKIPyAttentionBackend(unittest.TestCase):
    """Test the attention backend."""

    def test_get_name(self):
        from nkipy.vllm_plugin.attention import NKIPyAttentionBackend

        assert NKIPyAttentionBackend.get_name() == "CUSTOM"

    def test_get_kv_cache_shape(self):
        from nkipy.vllm_plugin.attention import NKIPyAttentionBackend

        shape = NKIPyAttentionBackend.get_kv_cache_shape(
            num_blocks=10, block_size=16, num_kv_heads=4, head_size=64
        )
        assert shape == (2, 10, 4, 16, 64)


class TestNKIPyModelRunner(unittest.TestCase):
    """Test the model runner."""

    def _make_vllm_config(self):
        cfg = MagicMock()
        cfg.model_config.model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        cfg.model_config.hf_config.vocab_size = 32000
        cfg.model_config.max_model_len = 128
        cfg.scheduler_config.max_num_seqs = 4
        cfg.cache_config.block_size = 16
        return cfg

    def test_init(self):
        from nkipy.vllm_plugin.model_runner import NKIPyModelRunner

        cfg = self._make_vllm_config()
        runner = NKIPyModelRunner(vllm_config=cfg)
        assert runner.model is None
        assert runner.vocab_size == 32000

    def test_get_kv_cache_spec_after_load(self):
        from nkipy.vllm_plugin.model_runner import NKIPyModelRunner
        from transformers import AutoConfig

        cfg = self._make_vllm_config()
        runner = NKIPyModelRunner(vllm_config=cfg)
        runner.hf_config = AutoConfig.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        specs = runner.get_kv_cache_spec()
        assert len(specs) == 22
        for name, spec in specs.items():
            assert spec.block_size == 16

    def test_execute_model_empty(self):
        from nkipy.vllm_plugin.model_runner import NKIPyModelRunner

        cfg = self._make_vllm_config()
        runner = NKIPyModelRunner(vllm_config=cfg)

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = set()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs.req_ids = []
        scheduler_output.total_num_scheduled_tokens = 0
        scheduler_output.num_scheduled_tokens = {}

        result = runner.execute_model(scheduler_output)
        assert result.req_ids == []


class TestNKIPyWorker(unittest.TestCase):
    """Test the worker."""

    def test_determine_available_memory(self):
        from nkipy.vllm_plugin.worker import NKIPyWorker

        cfg = MagicMock()
        worker = NKIPyWorker.__new__(NKIPyWorker)
        worker.cache_config = cfg.cache_config
        mem = NKIPyWorker.determine_available_memory(worker)
        assert mem == 1 * (1024 ** 3)

    @patch.dict(os.environ, {"NKIPY_HOST_STAGING": "0"})
    def test_prepare_push_skipped_without_host_staging(self):
        from nkipy.vllm_plugin.worker import NKIPyWorker

        worker = NKIPyWorker.__new__(NKIPyWorker)
        result = worker.nkipy_prepare_push()
        assert result == {"status": "skipped"}

    @patch.dict(os.environ, {"NKIPY_HOST_STAGING": "1"})
    def test_prepare_push_skipped_without_model(self):
        from nkipy.vllm_plugin.worker import NKIPyWorker

        worker = NKIPyWorker.__new__(NKIPyWorker)
        worker.model_runner = MagicMock()
        worker.model_runner._nkipy_model = None
        result = worker.nkipy_prepare_push()
        assert result == {"status": "skipped"}

    @patch.dict(os.environ, {"NKIPY_HOST_STAGING": "1"})
    def test_prepare_push_calls_start_pre_dma(self):
        from nkipy.vllm_plugin.worker import NKIPyWorker

        worker = NKIPyWorker.__new__(NKIPyWorker)
        worker.model_runner = MagicMock()
        mock_model = MagicMock()
        worker.model_runner._nkipy_model = mock_model

        with patch("relay.start_pre_dma_to_staging") as mock_pre_dma:
            result = worker.nkipy_prepare_push()
            mock_pre_dma.assert_called_once_with(mock_model)
            assert result == {"status": "preparing"}

    @patch.dict(os.environ, {"NKIPY_HOST_STAGING": "1"})
    def test_wake_up_sends_prepare_signal_rank0(self):
        """Rank 0 sends fire-and-forget prepare signal at wake_up start."""
        import time as _time
        from nkipy.vllm_plugin.worker import NKIPyWorker

        worker = NKIPyWorker.__new__(NKIPyWorker)
        worker._sleeping = True
        worker.rank = 0

        mock_post = MagicMock()
        with patch.dict("sys.modules", {"requests": MagicMock(post=mock_post)}):
            # The wake_up will fail early (no spike, no gloo) but the prepare
            # signal should have been submitted before any of that.
            try:
                worker.nkipy_wake_up("http://sender:8000")
            except Exception:
                pass

            # Give the background thread a moment to execute
            _time.sleep(0.2)
            mock_post.assert_called_once_with(
                "http://sender:8000/nkipy/p2p_prepare", timeout=5
            )

    @patch.dict(os.environ, {"NKIPY_HOST_STAGING": "1"})
    def test_wake_up_no_prepare_for_non_rank0(self):
        """Non-rank-0 workers should not send prepare signal."""
        from nkipy.vllm_plugin.worker import NKIPyWorker

        worker = NKIPyWorker.__new__(NKIPyWorker)
        worker._sleeping = True
        worker.rank = 5

        mock_post = MagicMock()
        with patch.dict("sys.modules", {"requests": MagicMock(post=mock_post)}):
            try:
                worker.nkipy_wake_up("http://sender:8000")
            except Exception:
                pass

            mock_post.assert_not_called()

    @patch.dict(os.environ, {"NKIPY_HOST_STAGING": "0"})
    def test_wake_up_no_prepare_without_host_staging(self):
        """Without NKIPY_HOST_STAGING=1, no prepare signal is sent."""
        from nkipy.vllm_plugin.worker import NKIPyWorker

        worker = NKIPyWorker.__new__(NKIPyWorker)
        worker._sleeping = True
        worker.rank = 0

        mock_post = MagicMock()
        with patch.dict("sys.modules", {"requests": MagicMock(post=mock_post)}):
            try:
                worker.nkipy_wake_up("http://sender:8000")
            except Exception:
                pass

            mock_post.assert_not_called()


class TestPreDmaStaging(unittest.TestCase):
    """Test start_pre_dma_to_staging early-exit logic."""

    @patch("relay.transfer.rank_endpoint")
    def test_noop_without_sender_staging(self, mock_ep):
        """No-op if _sender_staging is not set on rank_endpoint."""
        mock_ep._sender_staging = None
        # Should return before importing spike
        from relay.transfer import start_pre_dma_to_staging
        start_pre_dma_to_staging(MagicMock())

    @patch("relay.transfer.rank_endpoint")
    def test_noop_if_already_running(self, mock_ep):
        """No-op if a pre-DMA thread is already in progress."""
        mock_ep._sender_staging = MagicMock()
        mock_ep._pre_dma_thread = MagicMock()  # Already running
        from relay.transfer import start_pre_dma_to_staging
        start_pre_dma_to_staging(MagicMock())
        # _pre_dma_thread should remain the mock (not overwritten)
        assert mock_ep._pre_dma_thread is not None


if __name__ == "__main__":
    unittest.main()
