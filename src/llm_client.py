"""
llm_client.py - Claude/OpenAI abstraction for DeepAgent

Follows learn-claude-code pattern: simple, minimal abstraction.
Supports both Anthropic Claude and OpenAI-compatible APIs.

Usage:
    client = get_client()  # Uses LLM_PROVIDER env var
    response = client.chat(system, messages, tools)
"""

import os
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

# Load .env from project root
from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()  # Try cwd


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def chat(self, system: str, messages: list, tools: list) -> dict:
        """
        Send chat request to LLM.

        Args:
            system: System prompt
            messages: Conversation history
            tools: Tool definitions

        Returns:
            Dict with 'content', 'stop_reason', and optionally 'tool_calls'
        """
        pass


class AnthropicClient(LLMClient):
    """Claude API client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        base_url: Optional[str] = None,
    ):
        from anthropic import Anthropic

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")
        self.model = model

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = Anthropic(**client_kwargs)

    def chat(self, system: str, messages: list, tools: list) -> dict:
        """Call Claude API."""
        kwargs = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": 8000,
        }

        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)

        # Extract tool calls from content blocks
        tool_calls = []
        text_content = []

        for block in response.content:
            if hasattr(block, "type"):
                if block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    })
                elif block.type == "text":
                    text_content.append(block.text)

        return {
            "content": response.content,  # Keep raw content for message appending
            "text": "\n".join(text_content),
            "tool_calls": tool_calls,
            "stop_reason": response.stop_reason,
        }


class OpenAIClient(LLMClient):
    """OpenAI-compatible API client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
    ):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)

    def chat(self, system: str, messages: list, tools: list) -> dict:
        """Call OpenAI-compatible API."""
        # Convert Anthropic-style messages to OpenAI format
        openai_msgs = [{"role": "system", "content": system}]

        for msg in messages:
            if msg["role"] == "user":
                # Handle tool results
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "tool_result":
                            openai_msgs.append({
                                "role": "tool",
                                "tool_call_id": item["tool_use_id"],
                                "content": item["content"],
                            })
                else:
                    openai_msgs.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                # Handle assistant with tool calls
                content = msg.get("content")
                stored_tool_calls = msg.get("tool_calls", [])

                if isinstance(content, list):
                    # Anthropic format: content is list of blocks
                    text_parts = []
                    tool_calls = []
                    for block in content:
                        if hasattr(block, "type"):
                            if block.type == "text":
                                text_parts.append(block.text)
                            elif block.type == "tool_use":
                                tool_calls.append({
                                    "id": block.id,
                                    "type": "function",
                                    "function": {
                                        "name": block.name,
                                        "arguments": json.dumps(block.input),
                                    },
                                })
                    openai_msg = {"role": "assistant"}
                    text_content = "\n".join(text_parts)
                    if text_content:
                        openai_msg["content"] = text_content
                    if tool_calls:
                        openai_msg["tool_calls"] = tool_calls
                    openai_msgs.append(openai_msg)
                elif stored_tool_calls:
                    # OpenAI format: tool_calls stored separately
                    openai_msg = {"role": "assistant"}
                    if content:
                        openai_msg["content"] = content
                    # Convert stored tool_calls to OpenAI format
                    openai_msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("arguments", {})),
                            },
                        }
                        for tc in stored_tool_calls
                    ]
                    openai_msgs.append(openai_msg)
                else:
                    # Simple text content
                    openai_msgs.append({"role": "assistant", "content": content or ""})

        # Convert tools to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = []
            for t in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                    },
                })

        kwargs = {
            "model": self.model,
            "messages": openai_msgs,
            "max_tokens": 8000,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        # Normalize tool calls to common format
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {},
                })

        # Determine stop reason
        stop_reason = "end_turn"
        if choice.message.tool_calls:
            stop_reason = "tool_use"
        elif choice.finish_reason == "stop":
            stop_reason = "end_turn"

        return {
            "content": choice.message.content,
            "text": choice.message.content or "",
            "tool_calls": tool_calls,
            "stop_reason": stop_reason,
        }


def get_client(provider: Optional[str] = None, **kwargs) -> LLMClient:
    """
    Get LLM client based on provider.

    Args:
        provider: "anthropic" or "openai" (defaults to LLM_PROVIDER env var)
        **kwargs: Passed to client constructor

    Returns:
        LLMClient instance
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai")

    if provider.lower() == "openai":
        return OpenAIClient(**kwargs)
    else:
        return AnthropicClient(**kwargs)


if __name__ == "__main__":
    # Quick test
    client = get_client()
    print(f"Client type: {type(client).__name__}")
    print(f"Model: {client.model}")
