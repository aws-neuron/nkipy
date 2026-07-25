import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from nkipy_serving.tokenization.hf_tokenizer import HfTokenizer


def test_hf_tokenizer_roundtrip_local_path(tmp_path):
    tokenizer_dir = tmp_path / "hf_tok"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))

    tok = HfTokenizer(model_id=str(tokenizer_dir), local_files_only=True)
    ids = tok.encode("hello")
    assert isinstance(ids, np.ndarray)
    assert ids.dtype == np.int32
    assert ids.tolist() == [1]
    assert tok.decode(ids) == "hello"
