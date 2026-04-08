# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the NKIPy vLLM plugin components."""

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


if __name__ == "__main__":
    unittest.main()
