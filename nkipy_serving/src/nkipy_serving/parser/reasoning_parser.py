"""Reasoning extraction utilities.

This is intentionally lightweight and currently focuses on GPT-OSS Harmony
format (analysis/final channels).
"""

from __future__ import annotations

from dataclasses import dataclass

from nkipy_serving.parser.harmony_parser import HarmonyParser


@dataclass
class ParsedReasoning:
    reasoning_text: str
    normal_text: str


class GptOssReasoningParser:
    """Extract reasoning (analysis) vs normal (final/tool) text from Harmony output."""

    def __init__(self):
        self._parser = HarmonyParser()

    def parse_non_stream(self, full_text: str) -> ParsedReasoning:
        events = self._parser.parse(full_text)
        # Flush internal buffers for one-shot parsing.
        events += self._parser.parse("")

        reasoning = "".join(e.content for e in events if e.event_type == "reasoning")
        normal_parts: list[str] = []
        for e in events:
            if e.event_type == "normal":
                normal_parts.append(e.content)
            elif e.event_type == "tool_call":
                # Preserve structural markers if present (tool detectors may rely on them).
                normal_parts.append(e.raw_text if e.raw_text else e.content)
        return ParsedReasoning(
            reasoning_text=reasoning, normal_text="".join(normal_parts)
        )

    def parse_stream_chunk(self, delta_text: str) -> ParsedReasoning:
        events = self._parser.parse(delta_text)
        reasoning = "".join(e.content for e in events if e.event_type == "reasoning")
        normal_parts: list[str] = []
        for e in events:
            if e.event_type == "normal":
                normal_parts.append(e.content)
            elif e.event_type == "tool_call":
                normal_parts.append(e.raw_text if e.raw_text else e.content)
        return ParsedReasoning(
            reasoning_text=reasoning, normal_text="".join(normal_parts)
        )
