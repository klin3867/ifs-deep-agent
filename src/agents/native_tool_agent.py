"""
Native function calling agent - Claude Code style implementation.

This agent uses OpenAI's native function calling API instead of XML tag parsing,
resulting in cleaner code and more reliable tool execution.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator

from openai import AsyncOpenAI

from .schemas import SEARCH_TOOLS_SCHEMA, AgentEvent, ToolCall, StreamedResponse

logger = logging.getLogger(__name__)


class NativeToolAgent:
    """
    Claude Code-style agent with native OpenAI function calling.

    Key differences from XML-based approach:
    - Uses `tools=` parameter instead of XML tags in prompt
    - Tool results use `role: "tool"` with `tool_call_id`
    - Simple while loop without guard injections
    - Dynamic tool expansion via search_tools meta-tool
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        tool_manager,  # MCPToolCaller or ToolManager
        tool_retriever=None,  # MCPToolRetriever for semantic search
        memory_manager=None,
        system_prompt: str = "",
        max_iterations: int = 30,
    ):
        self.client = client
        self.model_name = model_name
        self.tool_manager = tool_manager
        self.tool_retriever = tool_retriever
        self.memory_manager = memory_manager
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

        # Start with just the meta-tool; expand dynamically
        self._active_tools: List[Dict] = [SEARCH_TOOLS_SCHEMA]
        self._discovered_tool_names: set = set()

        # Track tool usage for memory storage
        self._tools_used_this_run: List[Dict] = []

    async def run(
        self,
        user_message: str,
        conversation_history: List[Dict] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Execute agent loop with streaming events.

        Yields AgentEvent objects for UI updates:
        - thinking: Model is generating
        - tool_call: Tool invocation started
        - tool_result: Tool returned result
        - response: Final text response
        - done: Agent completed
        """
        messages = self._build_messages(user_message, conversation_history or [])

        # Reset per-run tracking
        self._tools_used_this_run = []

        for iteration in range(self.max_iterations):
            logger.info(f"Agent iteration {iteration + 1}/{self.max_iterations}")

            # 1. Call LLM with native function calling
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self._active_tools if self._active_tools else None,
                    tool_choice="auto",
                    stream=True,
                )
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                yield AgentEvent(type="error", data={"message": str(e)})
                return

            # 2. Accumulate streamed response
            accumulated = StreamedResponse()
            async for chunk in self._stream_response(response):
                if chunk.content:
                    yield AgentEvent(type="thinking", data={"content": chunk.content})
                accumulated = chunk

            # 3. No tool calls = done, return final response
            if not accumulated.tool_calls:
                if accumulated.content:
                    yield AgentEvent(type="response", data={"content": accumulated.content})
                    # Store memories on successful completion
                    await self._store_task_memories(user_message, accumulated.content)
                yield AgentEvent(type="done", data={})
                return

            # 4. Add assistant message to history
            messages.append(self._assistant_message(accumulated))

            # 5. Execute each tool call
            for tool_call in accumulated.tool_calls:
                yield AgentEvent(
                    type="tool_call",
                    data={
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                )

                # Execute the tool
                result = await self._execute_tool_call(tool_call)

                # Track tool usage for memory (skip meta-tools)
                if tool_call.name != "search_tools":
                    success = not (isinstance(result, dict) and "error" in result)
                    self._tools_used_this_run.append({
                        "tool_name": tool_call.name,
                        "arguments": tool_call.arguments,
                        "success": success,
                    })

                yield AgentEvent(
                    type="tool_result",
                    data={
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "result": result,
                    }
                )

                # Add tool result with proper role
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result) if isinstance(result, dict) else str(result),
                })

            # 6. Memory folding if enabled and context growing
            if self.memory_manager and len(messages) > 20:
                messages = await self._fold_memory(messages)

        # Max iterations reached
        yield AgentEvent(
            type="error",
            data={"message": f"Max iterations ({self.max_iterations}) reached"}
        )

    def _build_messages(
        self,
        user_message: str,
        history: List[Dict],
    ) -> List[Dict]:
        """Build message list with system prompt, relevant memories, and history."""
        messages = []

        # Build system prompt with relevant memories
        system_content = self.system_prompt or ""

        # Retrieve relevant past memories
        if self.memory_manager:
            memory_context = self._get_relevant_memories(user_message)
            if memory_context:
                system_content += f"\n\n## Relevant Past Experience\n{memory_context}"

        if system_content:
            messages.append({"role": "system", "content": system_content})

        # Conversation history
        messages.extend(history)

        # Current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def _get_relevant_memories(self, query: str) -> str:
        """
        Retrieve relevant episodic memories for the current task.

        Returns formatted string of relevant past experiences.
        """
        if not self.memory_manager:
            return ""

        try:
            # Use memory manager's retrieval (keyword-based)
            relevant = self.memory_manager.retrieve_relevant_episodic_memories(
                query=query,
                top_k=3,
            )

            if not relevant:
                return ""

            # Format memories for context
            memory_lines = []
            for mem in relevant:
                episode = mem.get("episode_memory", {})
                task_desc = episode.get("task_description", mem.get("task_description", ""))
                tools = episode.get("tools_called", [])
                success = episode.get("success", True)

                if task_desc:
                    status = "succeeded" if success else "had issues"
                    tools_str = ", ".join(tools[:5]) if tools else "none"
                    memory_lines.append(f"- Task: {task_desc[:100]}... (tools: {tools_str}, {status})")

            if memory_lines:
                return "\n".join(memory_lines)

        except Exception as e:
            logger.debug(f"Memory retrieval failed: {e}")

        return ""

    async def _stream_response(
        self,
        response,
    ) -> AsyncGenerator[StreamedResponse, None]:
        """
        Accumulate streaming response, handling tool_calls correctly.

        OpenAI streams tool_calls as deltas that need to be assembled.
        """
        accumulated = StreamedResponse()
        tool_calls_builder: Dict[int, Dict] = {}  # index -> {id, name, arguments_str}

        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            # Accumulate content
            if delta.content:
                accumulated.content += delta.content

            # Accumulate tool calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index

                    if idx not in tool_calls_builder:
                        tool_calls_builder[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }

                    if tc_delta.id:
                        tool_calls_builder[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_builder[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_builder[idx]["arguments"] += tc_delta.function.arguments

            # Check finish reason
            if chunk.choices[0].finish_reason:
                accumulated.finish_reason = chunk.choices[0].finish_reason

            yield accumulated

        # Parse accumulated tool calls
        for idx in sorted(tool_calls_builder.keys()):
            tc_data = tool_calls_builder[idx]
            try:
                args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
            except json.JSONDecodeError:
                args = {"_raw": tc_data["arguments"]}

            accumulated.tool_calls.append(ToolCall(
                id=tc_data["id"],
                name=tc_data["name"],
                arguments=args,
            ))

        yield accumulated

    async def _execute_tool_call(self, tool_call: ToolCall) -> Any:
        """Execute a single tool call."""

        # Handle search_tools meta-tool specially
        if tool_call.name == "search_tools":
            return await self._handle_search_tools(tool_call.arguments)

        # Regular tool execution via tool_manager
        try:
            # Format for tool_manager.call_tool()
            formatted_call = {
                "function": {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                }
            }
            result = await self.tool_manager.call_tool(formatted_call)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_call.name} - {e}")
            return {"error": str(e)}

    async def _handle_search_tools(self, arguments: Dict) -> Dict:
        """
        Handle search_tools meta-tool.

        Uses semantic retrieval to find relevant tools and expands
        the active tool set dynamically. Also retrieves domain knowledge
        from the memory manager's knowledge base.
        """
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 10)

        if not self.tool_retriever:
            return {"error": "Tool retriever not configured"}

        try:
            # Get tool names from retriever (returns List[str])
            tool_names = self.tool_retriever.retrieve(query, top_k=top_k)

            # Import TOOL_REGISTRY for knowledge
            from tools.mcp_tool_registry import TOOL_REGISTRY

            # Expand active tools with discovered ones
            new_tools = []
            knowledge_snippets = []

            for tool_name in tool_names:
                if tool_name and tool_name not in self._discovered_tool_names:
                    self._discovered_tool_names.add(tool_name)

                    # Get full schema from tool_manager
                    schema = self.tool_manager.get_tool_schema(tool_name)
                    if schema and "error" not in schema:
                        # Convert to OpenAI function format
                        new_tools.append({
                            "type": "function",
                            "function": {
                                "name": schema["name"],
                                "description": schema.get("description", ""),
                                "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
                            }
                        })

                # Collect knowledge from TOOL_REGISTRY
                if tool_name in TOOL_REGISTRY:
                    tool_info = TOOL_REGISTRY[tool_name]
                    knowledge_snippets.append(
                        f"**{tool_name}**: {tool_info.summary} (use when: {tool_info.use_when})"
                    )

            # Add new tools to active set
            self._active_tools.extend(new_tools)

            # =========================================================
            # MEMORY SYSTEM INTEGRATION
            # =========================================================
            # Retrieve procedural rules and domain knowledge from memory manager
            procedural_rules = []
            semantic_facts = []
            error_corrections = []

            if self.memory_manager:
                knowledge = self.memory_manager.retrieve_relevant_knowledge(
                    query=query,
                    tool_names=tool_names,
                    mcp_caller=self.tool_manager,
                    max_rules=10,
                    max_facts=5,
                )
                procedural_rules = knowledge.get('procedural_rules', [])
                semantic_facts = knowledge.get('semantic_facts', [])
                error_corrections = knowledge.get('error_corrections', [])

            return {
                "tools_found": len(tool_names),
                "tools_added": len(new_tools),
                "tool_names": tool_names,
                "knowledge": "\n".join(knowledge_snippets) if knowledge_snippets else None,
                # Memory system knowledge
                "procedural_rules": procedural_rules if procedural_rules else None,
                "semantic_facts": semantic_facts if semantic_facts else None,
                "error_corrections": error_corrections if error_corrections else None,
            }

        except Exception as e:
            logger.error(f"Tool search failed: {e}")
            return {"error": str(e)}

    def _assistant_message(self, response: StreamedResponse) -> Dict:
        """Convert StreamedResponse to assistant message dict."""
        msg = {"role": "assistant"}

        if response.content:
            msg["content"] = response.content

        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    }
                }
                for tc in response.tool_calls
            ]

        return msg

    async def _fold_memory(self, messages: List[Dict]) -> List[Dict]:
        """
        Compress conversation history using memory manager.

        This prevents context overflow on long conversations.
        """
        if not self.memory_manager:
            return messages

        try:
            # Keep system prompt and recent messages
            system_msg = messages[0] if messages[0]["role"] == "system" else None

            # Fold middle portion
            folded = await self.memory_manager.fold_context(messages)

            if system_msg:
                return [system_msg] + folded
            return folded

        except Exception as e:
            logger.warning(f"Memory folding failed: {e}")
            return messages

    def reset_tools(self):
        """Reset to initial state with only search_tools."""
        self._active_tools = [SEARCH_TOOLS_SCHEMA]
        self._discovered_tool_names.clear()
        self._tools_used_this_run = []

    async def _store_task_memories(self, user_message: str, final_response: str) -> None:
        """
        Store episodic and tool memories after successful task completion.

        This enables cross-task learning by recording:
        - What tools were used and whether they succeeded
        - The task description and outcome
        """
        if not self.memory_manager or not self._tools_used_this_run:
            return

        try:
            # Build tool memory
            tool_memory = {
                "tools_used": self._tools_used_this_run,
                "derived_rules": [],  # Could be populated by analyzing patterns
            }

            # Store tool memory
            self.memory_manager.store_tool_memory(
                tool_memory=tool_memory,
                task_description=user_message,
            )

            # Build episodic memory (simplified)
            episode_memory = {
                "task_description": user_message,
                "tools_called": [t["tool_name"] for t in self._tools_used_this_run],
                "success": all(t.get("success", True) for t in self._tools_used_this_run),
                "summary": final_response[:500] if final_response else "",
            }

            # Store episodic memory
            self.memory_manager.store_episodic_memory(
                episode_memory=episode_memory,
                task_description=user_message,
                dataset_name="mcp",
            )

            # Persist to disk
            self.memory_manager.save_memories()

            logger.info(f"Stored memories: {len(self._tools_used_this_run)} tools used")

        except Exception as e:
            logger.warning(f"Failed to store task memories: {e}")
