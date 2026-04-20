"""
LLMRouter: unified interface for OpenAI-compatible and Anthropic APIs.

Canonical message format (stored in history):
  {"role": "system",    "content": "..."}
  {"role": "user",      "content": "..."}
  {"role": "assistant", "content": "...", "tool_calls": [{"id": "...", "name": "...", "input": {...}}]}
  {"role": "tool",      "tool_call_id": "...", "content": "..."}

Tools are defined in OpenAI format; router converts to Anthropic format when needed.
"""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    finish_reason: str   # "stop" | "tool_calls"
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


class LLMRouter:
    PROVIDERS = ("openai", "anthropic")

    def __init__(self, provider: str, client, model: str):
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from: {self.PROVIDERS}")
        self.provider = provider
        self.client   = client
        self.model    = model

    # ── Public API ───────────────────────────────────────────────────────────

    def chat(self, *, messages: list, tools: list, system: str = "",
             max_tokens: int = 1024, temperature: float = 0.2) -> LLMResponse:
        """Send messages and return a normalised LLMResponse."""
        if self.provider == "openai":
            return self._chat_openai(messages, tools, system, max_tokens, temperature)
        return self._chat_anthropic(messages, tools, system, max_tokens, temperature)

    def make_assistant_message(self, response: LLMResponse) -> dict:
        """Build the canonical assistant message to append to history."""
        msg: dict = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "input": tc.input}
                for tc in response.tool_calls
            ]
        return msg

    def make_tool_result_message(self, tool_call_id: str, content: str) -> dict:
        """Build a canonical tool-result message."""
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    # ── OpenAI ───────────────────────────────────────────────────────────────

    def _chat_openai(self, messages, tools, system, max_tokens, temperature) -> LLMResponse:
        full = self._inject_system_openai(messages, system)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._to_openai_messages(full),
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = resp.choices[0]
        msg    = choice.message

        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, input=json.loads(tc.function.arguments))
            for tc in (msg.tool_calls or [])
        ]
        return LLMResponse(
            finish_reason="tool_calls" if choice.finish_reason == "tool_calls" else "stop",
            content=msg.content,
            tool_calls=tool_calls,
            raw=resp,
        )

    @staticmethod
    def _inject_system_openai(messages: list, system: str) -> list:
        if not system:
            return messages
        if messages and messages[0].get("role") == "system":
            return messages  # already present
        return [{"role": "system", "content": system}] + messages

    @staticmethod
    def _to_openai_messages(messages: list) -> list:
        """Convert canonical messages to OpenAI wire format."""
        result = []
        for m in messages:
            role = m["role"]
            if role in ("system", "user"):
                result.append({"role": role, "content": m["content"]})
            elif role == "assistant":
                msg: dict = {"role": "assistant", "content": m.get("content")}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["input"], ensure_ascii=False),
                            },
                        }
                        for tc in m["tool_calls"]
                    ]
                result.append(msg)
            elif role == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": m["tool_call_id"],
                    "content": m["content"],
                })
        return result

    # ── Anthropic ────────────────────────────────────────────────────────────

    def _chat_anthropic(self, messages, tools, system, max_tokens, temperature) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            messages=self._to_anthropic_messages(messages),
            tools=self._to_anthropic_tools(tools),
            max_tokens=max_tokens,
            temperature=temperature,
        )

        tool_calls   = []
        text_content = None
        for block in resp.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

        return LLMResponse(
            finish_reason="tool_calls" if resp.stop_reason == "tool_use" else "stop",
            content=text_content,
            tool_calls=tool_calls,
            raw=resp,
        )

    @staticmethod
    def _to_anthropic_messages(messages: list) -> list:
        """Convert canonical messages to Anthropic wire format.

        Consecutive tool-result messages are merged into one user message
        because Anthropic requires all results from a single turn in one block.
        """
        result = []
        i = 0
        while i < len(messages):
            m    = messages[i]
            role = m["role"]

            if role == "system":
                i += 1
                continue  # handled via separate system= param

            if role == "user":
                result.append({"role": "user", "content": m["content"]})
                i += 1

            elif role == "assistant":
                content = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    content.append({
                        "type":  "tool_use",
                        "id":    tc["id"],
                        "name":  tc["name"],
                        "input": tc["input"],
                    })
                result.append({"role": "assistant", "content": content})
                i += 1

            elif role == "tool":
                # Collect all consecutive tool results into one user message
                tool_results = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    t = messages[i]
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": t["tool_call_id"],
                        "content":     t["content"],
                    })
                    i += 1
                result.append({"role": "user", "content": tool_results})

            else:
                i += 1

        return result

    @staticmethod
    def _to_anthropic_tools(tools: list) -> list:
        """Convert OpenAI-format tools to Anthropic format."""
        return [
            {
                "name":         t["function"]["name"],
                "description":  t["function"].get("description", ""),
                "input_schema": t["function"]["parameters"],
            }
            for t in tools
        ]
