from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer


class HfTokenizer:
    """HF tokenizer wrapper with numpy token ID interface."""

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        local_files_only: bool = True,
    ):
        model_id = model_id.strip()
        if not model_id:
            raise RuntimeError("tokenizer model_id must be non-empty")

        model_path = Path(model_id)
        if model_path.exists():
            snapshot_path = model_path
        else:
            snapshot_path = Path(
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    local_files_only=local_files_only,
                )
            )

        tokenizer_json_path = snapshot_path / "tokenizer.json"
        if not tokenizer_json_path.exists():
            raise RuntimeError(
                "Missing tokenizer.json for tokenizer model: "
                f"{model_id} (resolved path: {snapshot_path})"
            )
        self._snapshot_path = snapshot_path
        self._tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
        # Lazily created transformers tokenizer for chat template rendering.
        self._hf_tokenizer = None

    def encode(self, text: str, add_special_tokens: bool = True) -> np.ndarray:
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return np.asarray(encoding.ids, dtype=np.int32)

    def decode(
        self,
        token_ids: np.ndarray | list[int],
        skip_special_tokens: bool = False,
    ) -> str:
        arr = np.asarray(token_ids, dtype=np.int32)
        if arr.size == 0:
            return ""
        return self._tokenizer.decode(
            arr.tolist(), skip_special_tokens=skip_special_tokens
        )

    def batch_decode(
        self,
        batch: list[np.ndarray | list[int]],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        return [
            self.decode(item, skip_special_tokens=skip_special_tokens) for item in batch
        ]

    def batch_encode(
        self,
        batch: list[str],
        add_special_tokens: bool = True,
    ) -> list[np.ndarray]:
        return [
            self.encode(text, add_special_tokens=add_special_tokens) for text in batch
        ]

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())

    @property
    def tokenizer_class(self) -> str:
        return type(self._tokenizer).__name__

    @property
    def tokenizer(self):
        """Return a transformers tokenizer for prompt formatting (chat templates).

        This is intentionally lazy so the runtime can stay lightweight for
        completions-only workloads.
        """
        if self._hf_tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "transformers is required for chat-template rendering. "
                    "Install transformers + jinja2 or use /v1/completions."
                ) from exc
            # `local_files_only=True` since snapshot path is already resolved.
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                str(self._snapshot_path),
                local_files_only=True,
            )
        return self._hf_tokenizer
