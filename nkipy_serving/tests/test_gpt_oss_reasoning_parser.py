import pytest

from nkipy_serving.parser.reasoning_parser import GptOssReasoningParser


def test_gpt_oss_reasoning_parser_non_stream_split() -> None:
    parser = GptOssReasoningParser()
    text = (
        "<|start|>assistant<|channel|>analysis<|message|>I should think.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Paris<|return|>"
    )
    parsed = parser.parse_non_stream(text)
    assert parsed.reasoning_text == "I should think."
    assert parsed.normal_text == "Paris"


def test_gpt_oss_reasoning_parser_non_stream_no_reasoning() -> None:
    parser = GptOssReasoningParser()
    text = "<|start|>assistant<|channel|>final<|message|>Paris<|return|>"
    parsed = parser.parse_non_stream(text)
    assert parsed.reasoning_text == ""
    assert parsed.normal_text == "Paris"


@pytest.mark.parametrize(
    "chunks",
    [
        [
            "<|start|>assistant<|channel|>analysis<|message|>Thi",
            "nking<|end|><|start|>assistant<|channel|>final<|message|>Par",
            "is<|return|>",
        ],
        [
            "<|sta",
            "rt|>assistant<|channel|>analysis<|mess",
            "age|>Thi",
            "nking<|end|><|start|>assistant<|channel|>final<|mes",
            "sage|>Pa",
            "ris<|return|>",
        ],
    ],
)
def test_gpt_oss_reasoning_parser_stream_handles_partial_tokens(
    chunks: list[str],
) -> None:
    parser = GptOssReasoningParser()
    reasoning_parts: list[str] = []
    normal_parts: list[str] = []
    for chunk in chunks:
        parsed = parser.parse_stream_chunk(chunk)
        reasoning_parts.append(parsed.reasoning_text)
        normal_parts.append(parsed.normal_text)
    assert "".join(reasoning_parts) == "Thinking"
    assert "".join(normal_parts) == "Paris"
