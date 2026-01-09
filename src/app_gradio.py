"""
Gradio Chat UI for DeepAgent with streaming thoughts display.
Run with: python src/app_gradio.py
"""
import asyncio
import json
import os
import re
import sys
import yaml
from argparse import Namespace
from typing import Generator
import threading

import gradio as gr

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mcp_tool_registry import build_tool_prompt, TOOL_REGISTRY
from tools.tool_manager import ToolManager
from prompts.prompts_deepagent import (
    BEGIN_TOOL_CALL, END_TOOL_CALL,
    BEGIN_TOOL_SEARCH, END_TOOL_SEARCH,
    BEGIN_TOOL_RESPONSE, END_TOOL_RESPONSE,
    FOLD_THOUGHT,
    get_episode_memory_instruction,
    get_working_memory_instruction,
    get_tool_memory_instruction,
)

# Global state
_tool_manager = None
_client = None
_args = None
_loop = None
_interaction_history = []  # Track tool calls for memory folding


def load_config(config_path: str = "./config/base_config.yaml") -> Namespace:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    args = Namespace(**config)
    args.dataset_name = 'mcp'
    args.use_openai_chat = True
    args.max_tokens = 16384
    args.temperature = 0.7
    args.top_p = 0.9
    args.repetition_penalty = 1.0
    args.top_k_sampling = 50
    return args


def get_event_loop():
    """Get or create event loop for async operations."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop


async def initialize():
    """Initialize tool manager and OpenAI client."""
    global _tool_manager, _client, _args
    
    if _args is None:
        _args = load_config()
    
    if _tool_manager is None:
        _tool_manager = await ToolManager.create(_args)
    
    if _client is None:
        from openai import AsyncOpenAI
        _client = AsyncOpenAI(
            api_key=_args.api_key,
            base_url=_args.base_url if hasattr(_args, 'base_url') else None
        )
    
    return _tool_manager, _client, _args


def extract_between(text: str, start: str, end: str) -> str:
    """Extract text between markers."""
    try:
        s = text.rindex(start) + len(start)
        e = text.rindex(end)
        return text[s:e].strip()
    except ValueError:
        return ""


def get_system_prompt():
    """Build system prompt with tools."""
    return f"""You are DeepAgent, a general reasoning agent with scalable toolsets.

When you need to search for tools, use:
{BEGIN_TOOL_SEARCH}
your search query here
{END_TOOL_SEARCH}

When you need to call a tool, use this exact format:
{BEGIN_TOOL_CALL}
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
{END_TOOL_CALL}

The system will execute the tool and provide results in:
{BEGIN_TOOL_RESPONSE}
... result ...
{END_TOOL_RESPONSE}

If your reasoning becomes too lengthy, you get stuck, or need to change direction, you can fold your thoughts:
{FOLD_THOUGHT}

The system will summarize your progress into structured memory (episodic, working, and tool memory) and let you continue with a fresh context.

Available tools:
{build_tool_prompt()}

Think step by step. Show your reasoning process. When you have the final answer, present it clearly."""


def clean_response_for_display(text: str) -> str:
    """Remove tool markers and clean up response for chat display."""
    # Remove tool search blocks
    text = re.sub(
        rf'{re.escape(BEGIN_TOOL_SEARCH)}.*?{re.escape(END_TOOL_SEARCH)}',
        '[Searching for tools...]',
        text,
        flags=re.DOTALL
    )
    
    # Remove fold thought markers
    text = text.replace(FOLD_THOUGHT, '[Memory folding...]')
    
    # Format tool call blocks nicely
    def format_tool_call(match):
        try:
            tool_json = json.loads(match.group(1))
            tool_name = tool_json.get("name", "unknown")
            return f"🔧 **Calling {tool_name}**"
        except:
            return "[Calling tool...]"
    
    text = re.sub(
        rf'{re.escape(BEGIN_TOOL_CALL)}(.*?){re.escape(END_TOOL_CALL)}',
        format_tool_call,
        text,
        flags=re.DOTALL
    )
    
    # Remove tool response markers
    text = re.sub(
        rf'{re.escape(BEGIN_TOOL_RESPONSE)}.*?{re.escape(END_TOOL_RESPONSE)}',
        '',
        text,
        flags=re.DOTALL
    )
    
    return text.strip()


async def run_thought_folding(client, args, question: str, current_output: str, 
                               interaction_history: list, available_tools: str = "") -> tuple:
    """
    Generate three types of memory in parallel: episode, working, and tool memory.
    This is the brain-inspired memory consolidation from the DeepAgent paper.
    """
    # Format previous thoughts as steps
    previous_thoughts = current_output.split("\n\n")
    previous_thoughts = [f"Step {i+1}: {step}" for i, step in enumerate(previous_thoughts) if step.strip()]
    previous_thoughts = "\n\n".join(previous_thoughts)
    
    # Extract tool call history
    tool_call_history = []
    for interaction in interaction_history:
        if "tool_call" in interaction:
            tool_call_history.append({
                "tool_call": interaction.get("tool_call", ""),
                "tool_response": interaction.get("tool_response", "")[:500]  # Truncate
            })
    tool_call_history_str = json.dumps(tool_call_history, indent=2) if tool_call_history else "No tool calls yet."
    
    async def generate_memory(prompt: str) -> str:
        """Generate a memory using the LLM."""
        try:
            response = await client.chat.completions.create(
                model=args.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1024,
                timeout=60,
            )
            return response.choices[0].message.content or "{}"
        except Exception as e:
            return f'{{"error": "{str(e)}"}}'
    
    # Generate all three memories in parallel
    episode_prompt = get_episode_memory_instruction(question, previous_thoughts, available_tools)
    working_prompt = get_working_memory_instruction(question, previous_thoughts, available_tools)
    tool_prompt = get_tool_memory_instruction(question, previous_thoughts, tool_call_history_str, available_tools)
    
    episode_mem, working_mem, tool_mem = await asyncio.gather(
        generate_memory(episode_prompt),
        generate_memory(working_prompt),
        generate_memory(tool_prompt)
    )
    
    return episode_mem, working_mem, tool_mem


def convert_history_for_api(history: list) -> list:
    """Convert Gradio history format to OpenAI messages format."""
    messages = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            # New Gradio 6.x format: {"role": ..., "content": ...}
            messages.append({"role": msg["role"], "content": msg["content"]})
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            # Old format: [user_msg, assistant_msg]
            if msg[0]:
                messages.append({"role": "user", "content": msg[0]})
            if msg[1]:
                messages.append({"role": "assistant", "content": msg[1]})
    return messages


def append_to_history(history: list, role: str, content: str) -> list:
    """Append a message to history in Gradio 6.x format."""
    return history + [{"role": role, "content": content}]


def process_message_sync(user_message: str, history: list) -> Generator:
    """
    Process a message synchronously with streaming updates.
    Yields: (thoughts_text, chat_history)
    History format: list of (user_msg, assistant_msg) tuples
    """
    loop = get_event_loop()
    
    # Initialize
    try:
        tool_manager, client, args = loop.run_until_complete(initialize())
    except Exception as e:
        error_msg = f"❌ Initialization error: {str(e)}"
        new_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": error_msg}]
        yield error_msg, new_history
        return
    
    # Build messages for API
    messages = [{"role": "system", "content": get_system_prompt()}]
    messages.extend(convert_history_for_api(history))
    messages.append({"role": "user", "content": user_message})
    
    thoughts = f"📝 User: {user_message}\n\n🤔 Thinking...\n"
    
    # Show user message with pending response
    new_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": "⏳ Thinking..."}]
    yield thoughts, new_history
    
    try:
        # Make non-streaming call first (simpler and more reliable)
        async def get_response():
            return await client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                max_completion_tokens=4096,
                timeout=120,
            )
        
        response = loop.run_until_complete(get_response())
        full_response = response.choices[0].message.content or ""
        
        # Update thoughts with the raw response
        thoughts += full_response
        new_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": "⏳ Processing..."}]
        yield thoughts, new_history
        
        # Check for tool calls
        if END_TOOL_CALL in full_response:
            tool_call_json = extract_between(full_response, BEGIN_TOOL_CALL, END_TOOL_CALL)
            
            try:
                tool_call = json.loads(tool_call_json)
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("arguments", {})
                
                thoughts += f"\n\n🔧 Executing tool: {tool_name}\n"
                thoughts += f"Args: {json.dumps(tool_args, indent=2)}\n"
                new_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": f"🔧 Calling {tool_name}..."}]
                yield thoughts, new_history
                
                # Execute tool
                adapted_call = {"function": {"name": tool_name, "arguments": tool_args}}
                
                async def call_tool():
                    return await tool_manager.call_tool(adapted_call, {})
                
                try:
                    print(f"[DEBUG] Calling tool: {tool_name}")
                    result = loop.run_until_complete(call_tool())
                    print(f"[DEBUG] Tool returned successfully")
                    result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
                    
                    # Track tool call for memory folding
                    _interaction_history.append({
                        "tool_call": f"{tool_name}({json.dumps(tool_args)})",
                        "tool_response": result_str[:500]
                    })
                    
                    thoughts += f"\n📋 Tool Result:\n{result_str[:2000]}{'...' if len(result_str) > 2000 else ''}\n"
                    yield thoughts, new_history
                    
                    print(f"[DEBUG] Getting follow-up response...")
                    # Get follow-up response with tool result
                    messages.append({"role": "assistant", "content": full_response})
                    messages.append({"role": "user", "content": f"{BEGIN_TOOL_RESPONSE}\n{result_str}\n{END_TOOL_RESPONSE}"})
                    
                    async def get_followup():
                        print(f"[DEBUG] Calling LLM for follow-up...")
                        try:
                            return await client.chat.completions.create(
                                model=args.model_name,
                                messages=messages,
                                max_completion_tokens=4096,
                                timeout=60,
                            )
                        except Exception as e:
                            print(f"[DEBUG] LLM follow-up error: {e}")
                            raise
                    
                    print(f"[DEBUG] About to run get_followup in event loop...")
                    followup = loop.run_until_complete(get_followup())
                    print(f"[DEBUG] Got follow-up response")
                    followup_text = followup.choices[0].message.content or ""
                    
                    thoughts += f"\n\n🔄 Follow-up:\n{followup_text}"
                    
                    # Clean response for display
                    display_response = clean_response_for_display(full_response + "\n\n" + followup_text)
                    new_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": display_response}]
                    yield thoughts, new_history
                    
                except Exception as e:
                    error_msg = f"❌ Tool Error: {str(e)}"
                    thoughts += f"\n{error_msg}"
                    new_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": error_msg}]
                    yield thoughts, new_history
                    
            except json.JSONDecodeError as e:
                thoughts += f"\n⚠️ Failed to parse tool call: {e}"
                display_response = clean_response_for_display(full_response)
                new_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": display_response}]
                yield thoughts, new_history
        
        # Check for tool search
        elif END_TOOL_SEARCH in full_response:
            search_query = extract_between(full_response, BEGIN_TOOL_SEARCH, END_TOOL_SEARCH)
            thoughts += f"\n\n🔍 Searching for tools: {search_query}\n"
            
            # Find matching tools
            matching_tools = []
            search_lower = search_query.lower()
            for tool_name, tool_info in TOOL_REGISTRY.items():
                if (search_lower in tool_name.lower() or 
                    search_lower in tool_info.summary.lower()):
                    matching_tools.append(f"- {tool_name}: {tool_info.summary}")
            
            if matching_tools:
                tool_results = "Found tools:\n" + "\n".join(matching_tools[:10])
            else:
                tool_results = "No matching tools found."
            
            thoughts += f"📋 Search Results:\n{tool_results}\n"
            
            display_response = clean_response_for_display(full_response)
            new_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": display_response}]
            yield thoughts, new_history
        
        # Check for memory folding
        elif FOLD_THOUGHT in full_response:
            thoughts += f"\n\n🧠 **Memory Folding Triggered**\n"
            thoughts += "Consolidating reasoning into structured memory...\n"
            new_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": "🧠 Folding thoughts..."}]
            yield thoughts, new_history
            
            # Run thought folding
            async def do_folding():
                return await run_thought_folding(
                    client=client,
                    args=args,
                    question=user_message,
                    current_output=full_response,
                    interaction_history=_interaction_history,
                    available_tools=build_tool_prompt()
                )
            
            episode_mem, working_mem, tool_mem = loop.run_until_complete(do_folding())
            
            # Display the memories
            thoughts += f"\n📚 **Episodic Memory** (key events & decisions):\n{episode_mem[:1000]}\n"
            thoughts += f"\n🎯 **Working Memory** (current state & next steps):\n{working_mem[:1000]}\n"
            thoughts += f"\n🔧 **Tool Memory** (learned patterns):\n{tool_mem[:1000]}\n"
            yield thoughts, new_history
            
            # Continue reasoning with memory context
            memory_context = f"""Memory of previous folded thoughts:

**Episodic Memory:**
{episode_mem}

**Working Memory:**
{working_mem}

**Tool Memory:**
{tool_mem}

Now continue your reasoning with this consolidated memory. What should we do next?"""
            
            messages.append({"role": "assistant", "content": full_response})
            messages.append({"role": "user", "content": memory_context})
            
            async def get_continuation():
                return await client.chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    max_completion_tokens=4096,
                    timeout=120,
                )
            
            continuation = loop.run_until_complete(get_continuation())
            continuation_text = continuation.choices[0].message.content or ""
            
            thoughts += f"\n\n🔄 **Continued Reasoning:**\n{continuation_text}"
            display_response = clean_response_for_display(continuation_text)
            new_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": display_response}]
            yield thoughts, new_history
        
        else:
            # No tool call, just return the response
            display_response = clean_response_for_display(full_response)
            new_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": display_response}]
            yield thoughts, new_history
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        thoughts += f"\n\n❌ Error: {str(e)}\n{error_details}"
        new_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": f"❌ Error: {str(e)}"}]
        yield thoughts, new_history


# CSS for the UI
CUSTOM_CSS = """
.thoughts-box textarea {
    font-family: 'Menlo', 'Monaco', 'Consolas', monospace !important;
    font-size: 12px !important;
    line-height: 1.4 !important;
    background: #1e1e1e !important;
    color: #d4d4d4 !important;
    border-radius: 8px !important;
}
.main-header {
    text-align: center;
    padding: 10px;
}
"""


def create_demo():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="DeepAgent", css=CUSTOM_CSS) as demo:
        
        with gr.Row():
            # Left panel - Raw Thoughts
            with gr.Column(scale=1):
                gr.Markdown("### 🧠 Raw Thoughts\n*Watch the agent's reasoning process*")
                thoughts_display = gr.Textbox(
                    label="",
                    lines=25,
                    max_lines=40,
                    interactive=False,
                    show_label=False,
                    elem_classes=["thoughts-box"],
                )
            
            # Right panel - Chat
            with gr.Column(scale=1):
                gr.Markdown("### 🤖 DeepAgent\n*A General Reasoning Agent with Scalable Toolsets*")
                
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=450,
                    show_label=False,
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a question or describe a task...",
                        show_label=False,
                        scale=9,
                        container=False,
                    )
                    submit_btn = gr.Button("🔍", scale=1, variant="primary")
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", size="sm")
        
        def respond(message, history):
            """Handle user message with streaming updates."""
            if not message or not message.strip():
                yield "", history, ""
                return
            
            # Stream the response
            for thoughts, new_history in process_message_sync(message.strip(), history):
                yield "", new_history, thoughts
        
        def clear_chat():
            global _interaction_history
            _interaction_history = []  # Clear tool interaction history
            return [], ""
        
        # Event handlers
        submit_btn.click(
            respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, thoughts_display],
        )
        
        msg_input.submit(
            respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, thoughts_display],
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, thoughts_display],
        )
        
        # Example queries
        gr.Examples(
            examples=[
                "What is 2 + 2?",
                "List the available MCP tools",
                "What customers have orders that are past due?",
            ],
            inputs=msg_input,
        )
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepAgent Gradio UI")
    parser.add_argument("--port", type=int, default=7865, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--config", default="./config/base_config.yaml", help="Config path")
    
    cli_args = parser.parse_args()
    
    print(f"🚀 Starting DeepAgent UI on http://127.0.0.1:{cli_args.port}")
    
    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=cli_args.port,
        share=cli_args.share,
    )
