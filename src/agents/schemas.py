"""
Schemas and types for native function calling agent.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# Meta-tool for dynamic tool discovery
SEARCH_TOOLS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_tools",
        "description": "Search for relevant tools to accomplish a task. Returns tool schemas and domain knowledge. Always call this first to discover available tools.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Description of what you want to accomplish (e.g., 'check inventory levels', 'create shipment')"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of tools to return (default: 10)",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    }
}


@dataclass
class AgentEvent:
    """Event emitted during agent execution for streaming UI."""
    type: Literal["thinking", "tool_call", "tool_result", "response", "error", "done"]
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "data": self.data}


@dataclass
class ToolCall:
    """Parsed tool call from model response."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class StreamedResponse:
    """Accumulated response from streaming completion."""
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
