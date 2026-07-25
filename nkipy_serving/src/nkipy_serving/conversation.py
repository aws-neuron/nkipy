"""Conversation helpers for chat-template style formatting."""

from __future__ import annotations

import copy
import json
from typing import Any

DSV4_BOS_TOKEN = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
DSV4_EOS_TOKEN = "<\uff5cend\u2581of\u2581sentence\uff5c>"
DSV4_THINKING_START_TOKEN = "<think>"
DSV4_THINKING_END_TOKEN = "</think>"
DSV4_USER_TOKEN = "<\uff5cUser\uff5c>"
DSV4_ASSISTANT_TOKEN = "<\uff5cAssistant\uff5c>"
DSV4_LATEST_REMINDER_TOKEN = "<\uff5clatest_reminder\uff5c>"
DSV4_DSML_TOKEN = "\uff5cDSML\uff5c"

_DSV4_TASK_TOKENS = {
    "action": "<\uff5caction\uff5c>",
    "query": "<\uff5cquery\uff5c>",
    "authority": "<\uff5cauthority\uff5c>",
    "domain": "<\uff5cdomain\uff5c>",
    "title": "<\uff5ctitle\uff5c>",
    "read_url": "<\uff5cread_url\uff5c>",
}

_DSV4_REASONING_EFFORT_MAX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
)

_DSV4_TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def _message_content(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n\n".join(part for part in parts if part)
    return str(content)


def _dsv4_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(value, ensure_ascii=True)


def _dsv4_find_last_user_index(messages: list[dict[str, Any]]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in {"user", "developer"}:
            return idx
    return -1


def _dsv4_tools_from_openai_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tool["function"] for tool in tools]


def _dsv4_tool_calls_from_openai_format(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"]["arguments"],
        }
        for tool_call in tool_calls
    ]


def _dsv4_render_tools(tools: list[dict[str, Any]]) -> str:
    return _DSV4_TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(_dsv4_json(t) for t in tools),
        dsml_token=DSV4_DSML_TOKEN,
        thinking_start_token=DSV4_THINKING_START_TOKEN,
        thinking_end_token=DSV4_THINKING_END_TOKEN,
    )


def _dsv4_encode_tool_arguments(tool_call: dict[str, Any]) -> str:
    template = '<{dsml_token}parameter name="{key}" string="{is_str}">{value}</{dsml_token}parameter>'
    raw_arguments = tool_call.get("arguments", {})
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {"arguments": raw_arguments}
    else:
        arguments = raw_arguments
    parts: list[str] = []
    for key, value in arguments.items():
        parts.append(
            template.format(
                dsml_token=DSV4_DSML_TOKEN,
                key=key,
                is_str="true" if isinstance(value, str) else "false",
                value=value if isinstance(value, str) else _dsv4_json(value),
            )
        )
    return "\n".join(parts)


def _dsv4_merge_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for msg in messages:
        msg = copy.deepcopy(msg)
        role = msg.get("role")
        if role == "tool":
            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
            ):
                merged[-1]["content_blocks"].append(tool_block)
            else:
                merged.append({"role": "user", "content_blocks": [tool_block]})
        elif role == "user":
            if isinstance(msg.get("content"), list):
                content_blocks = []
                for item in msg["content"]:
                    if isinstance(item, dict):
                        content_blocks.append(item)
                    else:
                        content_blocks.append({"type": "text", "text": str(item)})
            else:
                content_blocks = [{"type": "text", "text": _message_content(msg)}]
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
                and merged[-1].get("task") is None
            ):
                merged[-1]["content_blocks"].extend(content_blocks)
            else:
                new_msg = copy.deepcopy(msg)
                new_msg["content"] = _message_content(msg)
                new_msg["content_blocks"] = content_blocks
                merged.append(new_msg)
        else:
            merged.append(msg)
    return merged


def _dsv4_sort_tool_results_by_call_order(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    last_tool_call_order: dict[str, int] = {}
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            last_tool_call_order = {}
            for idx, tool_call in enumerate(msg["tool_calls"]):
                tool_id = tool_call.get("id") or tool_call.get("function", {}).get(
                    "id", ""
                )
                if tool_id:
                    last_tool_call_order[tool_id] = idx
        elif role == "user" and msg.get("content_blocks"):
            tool_blocks = [
                block
                for block in msg["content_blocks"]
                if block.get("type") == "tool_result"
            ]
            if len(tool_blocks) > 1 and last_tool_call_order:
                sorted_blocks = sorted(
                    tool_blocks,
                    key=lambda block: last_tool_call_order.get(
                        block.get("tool_use_id", ""), 0
                    ),
                )
                sorted_idx = 0
                new_blocks = []
                for block in msg["content_blocks"]:
                    if block.get("type") == "tool_result":
                        new_blocks.append(sorted_blocks[sorted_idx])
                        sorted_idx += 1
                    else:
                        new_blocks.append(block)
                msg["content_blocks"] = new_blocks
    return messages


def _dsv4_drop_thinking_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    last_user_idx = _dsv4_find_last_user_index(messages)
    result: list[dict[str, Any]] = []
    keep_roles = {"user", "system", "tool", "latest_reminder", "direct_search_results"}
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in keep_roles or idx >= last_user_idx:
            result.append(msg)
        elif role == "assistant":
            msg = copy.copy(msg)
            msg.pop("reasoning_content", None)
            result.append(msg)
    return result


def _dsv4_render_content_blocks(msg: dict[str, Any]) -> str:
    content_blocks = msg.get("content_blocks")
    if not content_blocks:
        return _message_content(msg)

    parts: list[str] = []
    for block in content_blocks:
        block_type = block.get("type")
        if block_type == "text":
            parts.append(str(block.get("text", "")))
        elif block_type == "tool_result":
            tool_content = block.get("content", "")
            if isinstance(tool_content, list):
                text_parts = []
                for item in tool_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                    else:
                        text_parts.append(
                            f"[Unsupported {item.get('type') if isinstance(item, dict) else type(item).__name__}]"
                        )
                tool_content = "\n\n".join(text_parts)
            parts.append(f"<tool_result>{tool_content}</tool_result>")
        else:
            parts.append(f"[Unsupported {block_type}]")
    return "\n\n".join(parts)


def _dsv4_render_message(
    index: int,
    messages: list[dict[str, Any]],
    *,
    thinking_mode: str,
    drop_thinking: bool,
    reasoning_effort: str | None,
) -> str:
    if thinking_mode not in {"chat", "thinking"}:
        raise ValueError(f"Invalid DeepSeek-V4 thinking_mode: {thinking_mode!r}")

    prompt = ""
    msg = messages[index]
    role = msg.get("role")
    content = _message_content(msg)
    tools = msg.get("tools")
    response_format = msg.get("response_format")
    tool_calls = msg.get("tool_calls")
    last_user_idx = _dsv4_find_last_user_index(messages)

    if tools:
        tools = _dsv4_tools_from_openai_format(tools)
    if tool_calls:
        tool_calls = _dsv4_tool_calls_from_openai_format(tool_calls)

    if reasoning_effort not in {"max", "high", None}:
        raise ValueError(f"Invalid DeepSeek-V4 reasoning effort: {reasoning_effort!r}")
    if index == 0 and thinking_mode == "thinking" and reasoning_effort == "max":
        prompt += _DSV4_REASONING_EFFORT_MAX

    if role == "system":
        prompt += content
        if tools:
            prompt += "\n\n" + _dsv4_render_tools(tools)
        if response_format:
            prompt += "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n"
            prompt += _dsv4_json(response_format)
    elif role == "developer":
        if not content:
            raise ValueError(f"Invalid message for role `{role}`: {msg}")
        developer_content = DSV4_USER_TOKEN + content
        if tools:
            developer_content += "\n\n" + _dsv4_render_tools(tools)
        if response_format:
            developer_content += "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n"
            developer_content += _dsv4_json(response_format)
        prompt += developer_content
    elif role == "user":
        prompt += DSV4_USER_TOKEN + _dsv4_render_content_blocks(msg)
    elif role == "latest_reminder":
        prompt += DSV4_LATEST_REMINDER_TOKEN + content
    elif role == "tool":
        raise NotImplementedError(
            "DeepSeek-V4 merges tool messages into user messages before rendering."
        )
    elif role == "assistant":
        tool_content = ""
        if tool_calls:
            rendered_calls = [
                (
                    f'<{DSV4_DSML_TOKEN}invoke name="{tool_call.get("name")}">\n'
                    f"{_dsv4_encode_tool_arguments(tool_call)}\n"
                    f"</{DSV4_DSML_TOKEN}invoke>"
                )
                for tool_call in tool_calls
            ]
            tool_content = (
                f"\n\n<{DSV4_DSML_TOKEN}tool_calls>\n"
                + "\n".join(rendered_calls)
                + f"\n</{DSV4_DSML_TOKEN}tool_calls>"
            )

        reasoning_content = str(
            msg.get("reasoning_content") or msg.get("reasoning") or ""
        )
        prev_has_task = index - 1 >= 0 and messages[index - 1].get("task") is not None
        thinking_part = ""
        if thinking_mode == "thinking" and not prev_has_task:
            if not drop_thinking or index > last_user_idx:
                thinking_part = reasoning_content + DSV4_THINKING_END_TOKEN
        prompt += thinking_part + content + tool_content
        if not msg.get("wo_eos", False):
            prompt += DSV4_EOS_TOKEN
    else:
        raise ValueError(f"Unsupported DeepSeek-V4 chat role: {role}")

    if index + 1 < len(messages) and messages[index + 1].get("role") not in {
        "assistant",
        "latest_reminder",
    }:
        return prompt

    task = msg.get("task")
    if task is not None:
        if task not in _DSV4_TASK_TOKENS:
            raise ValueError(
                f"Invalid DeepSeek-V4 task: {task!r}. Valid tasks: {list(_DSV4_TASK_TOKENS)}"
            )
        if task != "action":
            prompt += _DSV4_TASK_TOKENS[task]
        else:
            prompt += DSV4_ASSISTANT_TOKEN
            prompt += (
                DSV4_THINKING_START_TOKEN
                if thinking_mode == "thinking"
                else DSV4_THINKING_END_TOKEN
            )
            prompt += _DSV4_TASK_TOKENS[task]
    elif role in {"user", "developer"}:
        prompt += DSV4_ASSISTANT_TOKEN
        if thinking_mode == "thinking" and (
            not drop_thinking or index >= last_user_idx
        ):
            prompt += DSV4_THINKING_START_TOKEN
        else:
            prompt += DSV4_THINKING_END_TOKEN

    return prompt


def generate_deepseek_v4_chat_conv(
    messages: list[dict[str, Any]],
    *,
    thinking: bool = False,
    reasoning_effort: str | None = None,
) -> str:
    """Render OpenAI chat messages using the upstream DeepSeek-V4 encoding."""
    thinking_mode = "thinking" if thinking else "chat"
    context: list[dict[str, Any]] = []
    messages = _dsv4_merge_tool_messages(messages)
    messages = _dsv4_sort_tool_results_by_call_order(context + messages)[len(context) :]
    full_messages = context + messages

    drop_thinking = not any(m.get("tools") for m in full_messages)
    if thinking_mode == "thinking" and drop_thinking:
        full_messages = _dsv4_drop_thinking_messages(full_messages)
        num_to_render = len(full_messages) - len(_dsv4_drop_thinking_messages(context))
        context_len = len(full_messages) - num_to_render
    else:
        num_to_render = len(messages)
        context_len = len(context)

    prompt = DSV4_BOS_TOKEN
    for idx in range(num_to_render):
        prompt += _dsv4_render_message(
            idx + context_len,
            full_messages,
            thinking_mode=thinking_mode,
            drop_thinking=drop_thinking,
            reasoning_effort=reasoning_effort,
        )
    return prompt


def generate_chat_conv(
    messages: list[dict[str, Any]],
    tokenizer: Any = None,
    chat_template: str | None = None,
) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=chat_template,
        )

    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        lines.append(f"<|{role}|>\n{content}")
    lines.append("<|assistant|>")
    return "\n".join(lines)
