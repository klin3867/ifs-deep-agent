"""
agent.py - Claude Code-style agent for DeepAgent (MCP focus)

Core Philosophy: "The model is 80%. Code is 20%."
The model controls the loop - we just provide tools and stay out of the way.

Core loop:
    while True:
        response = model(messages, tools)
        if response.stop_reason != "tool_use":
            return response.text
        results = execute(response.tool_calls)
        messages.append(results)

Usage:
    agent = Agent.from_config("config/base_config.yaml")
    result = agent.run("What inventory do we have?")
"""

import json
import os
import sys
import asyncio
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from prompt_loader import PromptLoader
from llm_client import get_client, LLMClient

# Import MCP tools
try:
    from tools.mcp_client import MCPToolCaller
    from tools.mcp_tool_registry import MCPToolRetriever, get_tool_catalog, search_tools_by_keywords
    from tools.memory_manager import MemoryManager
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    MCPToolCaller = None
    MCPToolRetriever = None
    get_tool_catalog = None
    search_tools_by_keywords = None
    MemoryManager = None

# Knowledge base path
KNOWLEDGE_PATH = Path(__file__).parent.parent / "config" / "ifs_knowledge.yaml"


# =============================================================================
# Agent Configuration
# =============================================================================

AGENT_TYPES = {
    "Explore": {
        "system": "agent-prompt-explore.md",
        "tools": ["MCPSearch"],
    },
    "Plan": {
        "system": "agent-prompt-plan.md",
        "tools": ["MCPSearch", "TodoWrite", "AskUserQuestion"],
    },
    "general-purpose": {
        "system": "system-prompt-main.md",
        "tools": "*",
    },
    "summarizer": {
        "system": "agent-prompt-summarizer.md",
        "tools": [],
    },
}

ORCHESTRATION_TOOLS = {
    "MCPSearch": {
        "prompt": "tool-description-mcpsearch.md",
        "schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query or 'select:<tool_name>' to load a specific tool"
                }
            },
            "required": ["query"],
        },
    },
    "Task": {
        "prompt": "tool-description-task.md",
        "schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Task for subagent"},
                "subagent_type": {"type": "string", "enum": ["Explore", "Plan", "general-purpose"]},
            },
            "required": ["prompt", "subagent_type"],
        },
    },
    "TodoWrite": {
        "prompt": "tool-description-todowrite.md",
        "schema": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        },
                    },
                }
            },
            "required": ["todos"],
        },
    },
    "AskUserQuestion": {
        "prompt": "tool-description-askuserquestion.md",
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question to ask the user"},
            },
            "required": ["question"],
        },
    },
}


# =============================================================================
# TodoManager
# =============================================================================

class TodoManager:
    """Track tasks during agent execution."""

    def __init__(self):
        self.items = []

    def update(self, todos: list) -> str:
        """Update todo list."""
        self.items = todos
        completed = sum(1 for t in todos if t.get("status") == "completed")
        in_progress = sum(1 for t in todos if t.get("status") == "in_progress")
        pending = sum(1 for t in todos if t.get("status") == "pending")
        return f"Updated: {len(todos)} todos ({completed} done, {in_progress} in progress, {pending} pending)"


# =============================================================================
# Knowledge Injection
# =============================================================================

_knowledge_cache: dict = {}


def load_knowledge() -> dict:
    """Load IFS knowledge base (cached)."""
    global _knowledge_cache
    if _knowledge_cache:
        return _knowledge_cache

    if KNOWLEDGE_PATH.exists():
        import yaml
        with open(KNOWLEDGE_PATH) as f:
            _knowledge_cache = yaml.safe_load(f) or {}
    return _knowledge_cache


def get_tool_knowledge(tool_name: str) -> str:
    """Get procedural knowledge for a tool."""
    knowledge = load_knowledge()
    parts = []

    procedural = knowledge.get("procedural", {})
    if tool_name in procedural:
        rules = procedural[tool_name].get("rules", [])
        if rules:
            parts.append("**Rules:**")
            for rule in rules:
                parts.append(f"- {rule}")

    errors = knowledge.get("common_errors", [])
    for err in errors:
        keywords = err.get("keywords", [])
        if any(kw in tool_name.lower() for kw in keywords):
            parts.append(f"**Avoid:** {err.get('pattern', '')} → {err.get('correction', '')}")

    return "\n".join(parts) if parts else ""


# =============================================================================
# Context Management
# =============================================================================

MAX_CONTEXT_TOKENS = 100000
COMPACT_THRESHOLD = 0.75


def estimate_tokens(messages: list) -> int:
    """Rough token estimate: ~4 chars per token."""
    return sum(len(str(m)) for m in messages) // 4


# =============================================================================
# Agent
# =============================================================================

class Agent:
    """Claude Code-style agent with MCP tool support."""

    def __init__(
        self,
        prompt_loader: PromptLoader,
        llm: LLMClient,
        mcp: Optional["MCPToolCaller"] = None,
        memory_manager: Optional["MemoryManager"] = None,
        workdir: Optional[str] = None,
    ):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.mcp = mcp
        self.memory_manager = memory_manager
        self.workdir = Path(workdir or os.getcwd())
        self.todo = TodoManager()
        self._discovered_tools = []

        # MCP tool registry for semantic search
        self.retriever = None
        if HAS_MCP and MCPToolRetriever:
            try:
                self.retriever = MCPToolRetriever.get_instance()
            except Exception:
                pass

    def _build_system_prompt(self, agent_type: str) -> str:
        """Compose system prompt."""
        config = AGENT_TYPES.get(agent_type, AGENT_TYPES["general-purpose"])
        parts = []

        try:
            parts.append(self.prompt_loader.load(config["system"]))
        except FileNotFoundError:
            parts.append(f"You are an IFS Cloud ERP assistant ({agent_type} mode).")

        if get_tool_catalog and agent_type != "summarizer":
            parts.append(get_tool_catalog(compact=True))

        parts.append(f"\nWorking directory: {self.workdir}")

        return "\n\n".join(parts)

    def _build_tools(self, tool_names: list) -> list:
        """Build tools array."""
        if tool_names == "*":
            tool_names = list(ORCHESTRATION_TOOLS.keys())

        tools = []
        for name in tool_names:
            if name not in ORCHESTRATION_TOOLS:
                continue

            config = ORCHESTRATION_TOOLS[name]

            try:
                description = self.prompt_loader.load(config["prompt"])
            except FileNotFoundError:
                description = f"Tool: {name}"

            tools.append({
                "name": name,
                "description": description[:1000],
                "input_schema": config["schema"],
            })

        return tools

    def run(self, user_message: str, agent_type: str = "general-purpose") -> str:
        """
        Run agent loop until completion.

        Core pattern:
            while True:
                response = model(messages, tools)
                if response.stop_reason != "tool_use":
                    return response.text
                results = execute(response.tool_calls)
                messages.append(results)
        """
        config = AGENT_TYPES.get(agent_type, AGENT_TYPES["general-purpose"])
        system = self._build_system_prompt(agent_type)
        messages = [{"role": "user", "content": user_message}]

        base_tools = self._build_tools(config.get("tools", []))
        max_turns = 50

        for turn in range(max_turns):
            all_tools = base_tools + self._discovered_tools

            response = self.llm.chat(system, messages, all_tools)

            if response["stop_reason"] != "tool_use":
                return response.get("text", "")

            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                return response.get("text", "")

            results = []
            for tc in tool_calls:
                output = self._execute_tool(tc)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": output,
                })

            assistant_msg = {"role": "assistant", "content": response["content"]}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
            messages.append({"role": "user", "content": results})

            if self._should_compact(messages):
                messages = self._compact_messages(messages)

        return "Max turns reached."

    def _execute_tool(self, tc: dict) -> str:
        """Execute a tool call."""
        name = tc["name"]
        args = tc.get("arguments", {})

        print(f"\n> {name}: {args}")

        try:
            if name == "MCPSearch":
                result = self._handle_mcp_search(args.get("query", ""))
            elif name == "TodoWrite":
                result = self.todo.update(args.get("todos", []))
            elif name == "Task":
                result = self._spawn_subagent(args)
            elif name == "AskUserQuestion":
                result = self._ask_user(args.get("question", ""))
            elif self.mcp:
                result = asyncio.get_event_loop().run_until_complete(
                    self.mcp.call_tool({"function": {"name": name, "arguments": args}})
                )
                if isinstance(result, dict):
                    result = json.dumps(result, indent=2)
            else:
                result = f"Unknown tool: {name}"
        except Exception as e:
            result = f"Error: {e}"

        preview = result[:200] + "..." if len(result) > 200 else result
        print(f"  {preview}")

        return result

    def _handle_mcp_search(self, query: str) -> str:
        """MCPSearch: discover and load tools."""
        if query.startswith("select:"):
            tool_name = query[7:].strip()
            return self._load_tool_with_knowledge(tool_name)

        if search_tools_by_keywords:
            tools = search_tools_by_keywords(query, top_k=5)
            if tools:
                lines = ["**Found tools:**"]
                for t in tools:
                    flag = "!" if t.mutates else ""
                    lines.append(f"- {t.name}{flag}: {t.summary}")
                lines.append("\nUse `select:<tool_name>` to load a tool's schema.")
                return "\n".join(lines)

        if self.retriever:
            tool_names = self.retriever.retrieve(query, top_k=5)
            if tool_names:
                return f"Found tools: {', '.join(tool_names)}. Use `select:<name>` to load."

        return "No tools found matching query."

    def _load_tool_with_knowledge(self, tool_name: str) -> str:
        """Load tool schema and inject knowledge."""
        parts = [f"**Tool: {tool_name}**"]

        if self.mcp:
            schema = self.mcp.get_tool_schema(tool_name)
            if schema and "error" not in str(schema):
                self._discovered_tools.append(schema)
                params = schema.get("input_schema", {}).get("properties", {})
                if params:
                    parts.append("\n**Parameters:**")
                    for name, info in params.items():
                        desc = info.get("description", "")
                        parts.append(f"- {name}: {desc}")

        knowledge = get_tool_knowledge(tool_name)
        if knowledge:
            parts.append(f"\n{knowledge}")

        if len(parts) == 1:
            return f"Tool not found: {tool_name}"

        return "\n".join(parts)

    def _spawn_subagent(self, args: dict) -> str:
        """Task tool: spawn subagent."""
        prompt = args.get("prompt", "")
        subagent_type = args.get("subagent_type", "general-purpose")

        print(f"\n[Spawning {subagent_type} subagent]")

        subagent = Agent(
            prompt_loader=self.prompt_loader,
            llm=self.llm,
            mcp=self.mcp,
            memory_manager=self.memory_manager,
            workdir=str(self.workdir),
        )

        return subagent.run(prompt, subagent_type)

    def _ask_user(self, question: str) -> str:
        """Ask user for input."""
        print(f"\n? {question}")
        try:
            response = input("> ").strip()
            return response or "(no response)"
        except (EOFError, KeyboardInterrupt):
            return "(cancelled)"

    def _should_compact(self, messages: list) -> bool:
        """Check if conversation needs compaction."""
        tokens = estimate_tokens(messages)
        return tokens > MAX_CONTEXT_TOKENS * COMPACT_THRESHOLD

    def _compact_messages(self, messages: list) -> list:
        """Summarize conversation using summarizer subagent."""
        print("\n[Context compaction triggered]")

        conv_text = "\n".join(str(m) for m in messages[:-1])

        summary = self._spawn_subagent({
            "prompt": conv_text,
            "subagent_type": "summarizer"
        })

        last_msg = messages[-1] if messages else {"role": "user", "content": ""}
        return [
            {"role": "user", "content": f"<summary>\n{summary}\n</summary>"},
            last_msg
        ]

    @classmethod
    def from_config(cls, config_path: str) -> "Agent":
        """Create agent from config file."""
        import yaml

        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Prompts directory
        prompts_dir = config.get("prompts_dir", "./src/prompts")
        if not Path(prompts_dir).is_absolute():
            prompts_dir = config_path.parent / prompts_dir

        variables = config.get("variables", {})
        prompt_loader = PromptLoader(str(prompts_dir), variables)

        # LLM client
        provider = config.get("llm_provider", os.getenv("LLM_PROVIDER", "openai"))
        model = config.get("model_name") or config.get(f"{provider}_model")
        api_key = config.get("api_key")
        base_url = config.get("base_url")

        llm = get_client(provider, api_key=api_key, model=model, base_url=base_url)

        # MCP client
        mcp = None
        if HAS_MCP and MCPToolCaller:
            planning_url = config.get("mcp_planning_url", "http://localhost:8000/sse")
            customer_url = config.get("mcp_customer_url", "http://localhost:8001/sse")
            try:
                mcp = MCPToolCaller(planning_url=planning_url, customer_url=customer_url)
                asyncio.get_event_loop().run_until_complete(mcp.initialize())
                print(f"MCP: Connected to {len(mcp._tools)} tools")
            except Exception as e:
                print(f"MCP: Connection failed - {e}")
                mcp = None

        # Memory manager
        memory_manager = None
        if HAS_MCP and MemoryManager and config.get("memory_enabled", True):
            memory_dir = config.get("memory_cache_dir", "./cache/memory")
            memory_manager = MemoryManager(memory_dir=memory_dir)
            memory_manager.load_knowledge_base()

        return cls(
            prompt_loader=prompt_loader,
            llm=llm,
            mcp=mcp,
            memory_manager=memory_manager,
            workdir=config.get("workdir"),
        )


# =============================================================================
# Main REPL
# =============================================================================

def main():
    """Simple Read-Eval-Print Loop."""
    import argparse

    parser = argparse.ArgumentParser(description="DeepAgent - Claude Code Style")
    parser.add_argument("--config", default="config/base_config.yaml", help="Config file")
    parser.add_argument("--prompt", help="Single prompt (non-interactive)")
    parser.add_argument("--agent-type", default="general-purpose", help="Agent type")
    args = parser.parse_args()

    try:
        agent = Agent.from_config(args.config)
    except FileNotFoundError:
        print(f"Config not found: {args.config}, using defaults")
        prompt_loader = PromptLoader("./src/prompts")
        llm = get_client()
        agent = Agent(prompt_loader, llm)

    print(f"DeepAgent - {agent.workdir}")
    print(f"LLM: {type(agent.llm).__name__}")
    print("Type 'exit' to quit.\n")

    if args.prompt:
        result = agent.run(args.prompt, args.agent_type)
        print(result)
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        try:
            result = agent.run(user_input, args.agent_type)
            print(f"\nAssistant: {result}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
