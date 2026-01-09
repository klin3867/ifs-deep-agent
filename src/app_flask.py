"""
Flask Chat UI for DeepAgent with MCP integration.
Features: Streaming responses, thinking/reasoning panel, multi-step tool calls.
Run with: python src/app_flask.py
"""
import asyncio
import json
import os
import re
import sys
import yaml
import queue
import threading
from argparse import Namespace
from flask import Flask, render_template_string, request, jsonify, Response

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mcp_tool_registry import (
    build_tool_prompt, TOOL_REGISTRY, get_tools_for_intent,
    CATEGORY_DESCRIPTION, MCPToolRetriever, get_full_tool_schemas,
)
from tools.tool_manager import ToolManager
from prompts.prompts_deepagent import (
    BEGIN_TOOL_CALL, END_TOOL_CALL,
    BEGIN_TOOL_RESPONSE, END_TOOL_RESPONSE,
    BEGIN_TOOL_SEARCH, END_TOOL_SEARCH,
    BEGIN_TOOL_SEARCH_RESULT, END_TOOL_SEARCH_RESULT,
    FOLD_THOUGHT,
    get_episode_memory_instruction,
    get_working_memory_instruction,
    get_tool_memory_instruction,
    get_tool_call_intent_instruction,
    get_folded_thought_instruction,
)
from typing import Dict, List, Optional, Tuple

app = Flask(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


# Dev-only: opt-in raw thought streaming/display.
SHOW_RAW_THOUGHTS = _env_flag("DEEPAGENT_SHOW_RAW_THOUGHTS", default=False)

# Memory folding configuration
FOLD_ITERATION_THRESHOLD = 5  # Trigger fold after this many iterations
MAX_FOLDS_PER_REQUEST = 2     # Maximum folds per user request

# Task completion configuration (ReAct + Plan/Track/Recover)
ENABLE_TASK_PLANNING = _env_flag("DEEPAGENT_TASK_PLANNING", default=True)
ENABLE_INTENT_TRACKING = _env_flag("DEEPAGENT_INTENT_TRACKING", default=True)
ENABLE_COMPLETION_CHECK = _env_flag("DEEPAGENT_COMPLETION_CHECK", default=True)

# Multi-step workflow definitions
MULTI_STEP_WORKFLOWS = {
    'shipment': {
        'keywords': ['shipment', 'ship', 'move', 'transfer'],
        'tools': ['create_shipment_order', 'add_shipment_order_line', 'release_shipment_order'],
    },
    'reservation': {
        'keywords': ['reserve', 'reservation'],
        'tools': ['create_reservation', 'add_reservation_line', 'confirm_reservation'],
    },
}

# Global state
_tool_manager = None
_client = None
_aux_client = None  # Auxiliary client for memory folding
_args = None
_conversation_history = []


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


async def initialize():
    """Initialize tool manager and OpenAI client."""
    global _tool_manager, _client, _aux_client, _args

    if _args is None:
        _args = load_config()

    if _tool_manager is None:
        _tool_manager = await ToolManager.create(_args)
        # Load knowledge base (procedural rules + semantic facts) on first init
        if _tool_manager.memory_manager is not None:
            knowledge_loaded = _tool_manager.memory_manager.load_knowledge_base()
            if knowledge_loaded > 0:
                print(f"  📚 Loaded {knowledge_loaded} knowledge entries from config")

    if _client is None:
        from openai import AsyncOpenAI
        _client = AsyncOpenAI(
            api_key=_args.api_key,
            base_url=_args.base_url if hasattr(_args, 'base_url') else None
        )

    # Auxiliary client for memory folding (uses same config as main client)
    if _aux_client is None:
        from openai import AsyncOpenAI
        _aux_client = AsyncOpenAI(
            api_key=_args.api_key,
            base_url=_args.base_url if hasattr(_args, 'base_url') else None
        )

    return _tool_manager, _client, _args


def _extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response text."""
    if not text:
        return None
    # Try to find JSON in code block
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try to parse entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object in text
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


async def generate_memory_fold(
    user_message: str,
    reasoning_history: str,
    tool_call_history: List[Dict],
) -> Tuple[Dict, Dict, Dict]:
    """
    Generate brain-inspired memories from reasoning history.

    This implements the same memory folding mechanism as run_deep_agent.py,
    generating all three memory types (episodic, working, tool) in parallel.

    Args:
        user_message: The original user request
        reasoning_history: Accumulated reasoning text from the conversation
        tool_call_history: List of tool calls and their responses

    Returns:
        Tuple of (episode_memory, working_memory, tool_memory) dicts
    """
    global _aux_client, _args

    available_tools = build_tool_prompt()
    tool_history_str = json.dumps(tool_call_history, indent=2) if tool_call_history else "[]"

    # Get the model to use (prefer aux model if configured, otherwise use main model)
    model_name = getattr(_args, 'aux_model_name', None) or _args.model_name

    async def gen_episode():
        prompt = get_episode_memory_instruction(user_message, reasoning_history, available_tools)
        try:
            resp = await _aux_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
            return _extract_json_from_text(resp.choices[0].message.content)
        except Exception as e:
            print(f"Episode memory generation failed: {e}")
            return None

    async def gen_working():
        prompt = get_working_memory_instruction(user_message, reasoning_history, available_tools)
        try:
            resp = await _aux_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
            return _extract_json_from_text(resp.choices[0].message.content)
        except Exception as e:
            print(f"Working memory generation failed: {e}")
            return None

    async def gen_tool():
        prompt = get_tool_memory_instruction(user_message, reasoning_history, tool_history_str, available_tools)
        try:
            resp = await _aux_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
            return _extract_json_from_text(resp.choices[0].message.content)
        except Exception as e:
            print(f"Tool memory generation failed: {e}")
            return None

    # Generate all 3 memories in parallel
    episode, working, tool = await asyncio.gather(gen_episode(), gen_working(), gen_tool())
    return episode or {}, working or {}, tool or {}


def detect_workflow(user_message: str) -> Optional[Tuple[str, List[str]]]:
    """
    Detect if user message indicates a multi-step workflow.

    Returns:
        Tuple of (workflow_name, expected_tools) if detected, None otherwise.
    """
    user_lower = user_message.lower()
    for workflow_name, config in MULTI_STEP_WORKFLOWS.items():
        if any(kw in user_lower for kw in config['keywords']):
            return workflow_name, config['tools']
    return None


async def generate_task_plan(user_message: str) -> Optional[Dict]:
    """
    Generate working memory with next_actions before starting a multi-step task.
    Uses get_working_memory_instruction to create a checklist.

    Returns:
        Parsed JSON with immediate_goal and next_actions, or None on failure.
    """
    global _aux_client, _args

    if not ENABLE_TASK_PLANNING:
        return None

    available_tools = build_tool_prompt()
    model_name = getattr(_args, 'aux_model_name', None) or _args.model_name

    prompt = get_working_memory_instruction(
        question=user_message,
        prev_reasoning="",  # No reasoning yet at task start
        available_tools=available_tools
    )

    try:
        resp = await _aux_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3,
        )
        return _extract_json_from_text(resp.choices[0].message.content)
    except Exception as e:
        print(f"Task plan generation failed: {e}")
        return None


async def extract_tool_intent(reasoning_so_far: str) -> Optional[str]:
    """
    Extract the intent behind the most recent tool call to maintain progress awareness.

    Returns:
        Intent string (e.g., "Step 1 of 3: Create shipment header"), or None on failure.
    """
    global _aux_client, _args

    if not ENABLE_INTENT_TRACKING:
        return None

    model_name = getattr(_args, 'aux_model_name', None) or _args.model_name
    prompt = get_tool_call_intent_instruction(reasoning_so_far)

    try:
        resp = await _aux_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Tool intent extraction failed: {e}")
        return None


async def check_task_completion(
    user_message: str,
    reasoning_history: str,
    tool_history: List[str],
    expected_tools: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Check if a multi-step task is complete. If not, generate a recovery prompt.

    Args:
        user_message: Original user request
        reasoning_history: Accumulated reasoning text
        tool_history: List of tool names that have been called
        expected_tools: Expected tool sequence for the workflow

    Returns:
        Tuple of (is_complete, recovery_summary). If complete, summary is empty.
    """
    global _aux_client, _args

    if not ENABLE_COMPLETION_CHECK:
        return True, ""

    # If we don't have expected tools, assume complete
    if not expected_tools:
        return True, ""

    # Check if all expected tools have been called
    tools_called = set(tool_history)
    tools_expected = set(expected_tools)
    missing_tools = tools_expected - tools_called

    if not missing_tools:
        return True, ""

    # Task is incomplete - generate recovery prompt
    model_name = getattr(_args, 'aux_model_name', None) or _args.model_name
    prompt = get_folded_thought_instruction(user_message, reasoning_history)

    try:
        resp = await _aux_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        summary = resp.choices[0].message.content.strip()
        return False, summary
    except Exception as e:
        print(f"Task completion check failed: {e}")
        # On failure, provide a basic reminder
        return False, f"Task incomplete. Missing steps: {', '.join(missing_tools)}"


def extract_chain_of_thought(text: str) -> List[Dict[str, str]]:
    """
    Extract LLM reasoning text for display in Activity Panel.

    Captures any reasoning that appears BEFORE tool calls, tool searches, or fold markers.
    This is the actual "chain of thought" the LLM produces.

    Returns list of {"type": "reasoning", "content": "..."} dicts.
    """
    cot_items = []

    # Markers that indicate end of reasoning and start of action
    # Use regex patterns to handle malformed tags like <tool_call garbage>
    action_patterns = [
        r'<tool_call\b',   # Handles both <tool_call> and <tool_call garbage>
        r'<tool_search\b',
        FOLD_THOUGHT,
    ]

    # Find the earliest action marker
    earliest_pos = len(text)
    for pattern in action_patterns:
        match = re.search(pattern, text)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()

    # Extract reasoning before the first action
    if earliest_pos > 0:
        reasoning = text[:earliest_pos].strip()
        # Clean up: remove excessive whitespace, keep it readable
        reasoning = re.sub(r'\n{3,}', '\n\n', reasoning)  # Max 2 newlines
        reasoning = reasoning.strip()

        # Only include if there's substantial content (not just "I'll" or "Let me")
        if len(reasoning) > 30:
            # Truncate for display but keep enough context
            if len(reasoning) > 400:
                reasoning = reasoning[:400] + "..."
            cot_items.append({"type": "reasoning", "content": reasoning})

    # Also extract any reasoning BETWEEN tool responses and next tool calls
    # Look for text after END_TOOL_RESPONSE that precedes next action
    search_start = 0
    while True:
        resp_end = text.find(END_TOOL_RESPONSE, search_start)
        if resp_end == -1:
            break
        resp_end += len(END_TOOL_RESPONSE)

        # Find next action marker after this response
        next_action_pos = len(text)
        remaining_text = text[resp_end:]
        for pattern in action_patterns:
            match = re.search(pattern, remaining_text)
            if match:
                pos = resp_end + match.start()
                if pos < next_action_pos:
                    next_action_pos = pos

        # Extract reasoning between response and next action
        if next_action_pos > resp_end:
            between_text = text[resp_end:next_action_pos].strip()
            if len(between_text) > 30:
                if len(between_text) > 400:
                    between_text = between_text[:400] + "..."
                cot_items.append({"type": "reasoning", "content": between_text})

        search_start = resp_end

    return cot_items


def extract_between(text: str, start: str, end: str) -> str:
    """Extract text between markers."""
    try:
        s = text.rindex(start) + len(start)
        e = text.rindex(end)
        return text[s:e].strip()
    except ValueError:
        return ""


def extract_all_between(text: str, start: str, end: str) -> list[str]:
    """Extract all non-overlapping substrings between markers, in order."""
    results: list[str] = []
    i = 0
    while True:
        s = text.find(start, i)
        if s == -1:
            break
        s += len(start)
        e = text.find(end, s)
        if e == -1:
            break
        results.append(text[s:e].strip())
        i = e + len(end)
    return results


def has_tool_call_marker(text: str) -> bool:
    """Check if text contains a tool_call tag (exact or malformed)."""
    # Exact match
    if BEGIN_TOOL_CALL in text:
        return True
    # Flexible match for malformed tags like <tool_call garbage>
    if re.search(r'<tool_call\b', text):
        return True
    return False


def extract_tool_calls_robust(text: str) -> list[str]:
    """
    Robustly extract tool call JSON from LLM output, handling variations.

    The LLM sometimes outputs malformed tags like:
    - <tool_call garbage_here>{"name": ...}</tool_call>
    - <tool_call>{"name": ...}</tool_call garbage>

    This function handles these cases by finding JSON objects between
    flexible tag patterns.
    """
    results: list[str] = []

    # First try exact markers
    exact_results = extract_all_between(text, BEGIN_TOOL_CALL, END_TOOL_CALL)
    if exact_results:
        return exact_results

    # Fallback: Use regex to find <tool_call...>JSON</tool_call...>
    # This handles cases where LLM adds garbage after the tag name
    pattern = r'<tool_call[^>]*>\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\s*</tool_call[^>]*>'
    for match in re.finditer(pattern, text, re.DOTALL):
        json_str = match.group(1).strip()
        if json_str:
            results.append(json_str)

    if results:
        return results

    # Last resort: Just find JSON objects that look like tool calls
    # Look for {"name": "...", "arguments": ...} pattern
    json_pattern = r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        results.append(match.group(0))

    return results


def _extract_site_codes(user_text: str) -> list[str]:
    """Best-effort extraction of site codes from user text.

    Matches: ALL-CAPS alpha (2-4 chars) or numeric (2-4 digits) site codes.
    Examples: AC, 300, 400, MAIN
    """
    # Avoid interpreting quantities like "100,000" or "100, 000" as site codes
    # (it would otherwise match "100" and "000"). Strip comma-separated thousands
    # before extracting numeric site codes.
    text_no_thousands = re.sub(r"\b\d{1,3}(?:\s*,\s*\d{3})+\b", " ", user_text)

    # Match uppercase letters (2-4) or digits (2-4)
    alpha_candidates = re.findall(r"\b[A-Z]{2,4}\b", user_text)
    numeric_candidates = re.findall(r"\b\d{2,4}\b", text_no_thousands)

    blacklist = {"IFS", "ERP", "MCP", "JSON", "API"}
    seen: set[str] = set()
    sites: list[str] = []

    for token in alpha_candidates + numeric_candidates:
        if token in blacklist:
            continue
        if token not in seen:
            seen.add(token)
            sites.append(token)
    return sites


def _fmt_qty(value) -> str:
    try:
        f = float(value)
    except Exception:
        return "0"
    if abs(f - round(f)) < 1e-9:
        return f"{int(round(f)):,}"
    return f"{f:,.3f}"


def _summarize_inventory_by_site(inventory_results_by_site: dict[str, dict]) -> str:
    rows: list[tuple[str, int, float, float, float]] = []
    for site, result in inventory_results_by_site.items():
        data = (result or {}).get("data") or {}
        locations = data.get("locations") or []
        count = data.get("count") if isinstance(data.get("count"), int) else len(locations)
        totals = data.get("totals") or {}
        if totals:
            onhand = float(totals.get("onhand") or 0.0)
            reserved = float(totals.get("reserved") or 0.0)
            available = float(totals.get("available") or 0.0)
        else:
            onhand = float(sum(float(l.get("qty_onhand") or 0.0) for l in locations))
            reserved = float(sum(float(l.get("qty_reserved") or 0.0) for l in locations))
            available = float(sum(float(l.get("qty_available") or 0.0) for l in locations))
        rows.append((site, int(count), onhand, reserved, available))

    rows.sort(key=lambda r: r[0])
    lines = [
        "| Site | Locations | On hand | Reserved | Available |",
        "|---|---:|---:|---:|---:|",
    ]
    for site, count, onhand, reserved, available in rows:
        lines.append(
            f"| {site} | {count} | {_fmt_qty(onhand)} | {_fmt_qty(reserved)} | {_fmt_qty(available)} |"
        )
    return "\n".join(lines)


# Build system prompt with dynamic tool search
SYSTEM_PROMPT = f"""You are an autonomous assistant with access to IFS Cloud ERP tools via MCP.

## TOOL & KNOWLEDGE DISCOVERY (ALWAYS USE FIRST!)
Before calling a tool OR answering questions about IFS Cloud, search for relevant knowledge:
{BEGIN_TOOL_SEARCH}your search query (e.g., "create shipment", "warehouse site 105"){END_TOOL_SEARCH}

The system returns tools AND domain-specific knowledge:
{BEGIN_TOOL_SEARCH_RESULT}
{{"tools": [...], "knowledge": {{"rules": [...], "facts": [...]}}}}
{END_TOOL_SEARCH_RESULT}

CRITICAL: The "knowledge" section contains domain-specific rules that may CONTRADICT your training data.
Always trust the injected knowledge over your general knowledge about IFS Cloud.

## TOOL CALLING
Once you have the schema, call tools with exact parameter names:
{BEGIN_TOOL_CALL}
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
{END_TOOL_CALL}

The system executes and returns:
{BEGIN_TOOL_RESPONSE}
... result ...
{END_TOOL_RESPONSE}

## THOUGHT FOLDING
For complex multi-step tasks, you can compress your reasoning:
{FOLD_THOUGHT}

## AVAILABLE TOOL CATEGORIES
{CATEGORY_DESCRIPTION}

## IMPORTANT BEHAVIOR GUIDELINES
- Be AUTONOMOUS: Execute tasks without asking for confirmation.
- ALWAYS use {BEGIN_TOOL_SEARCH} FIRST - even for explanatory questions about IFS Cloud.
- The knowledge returned contains CUSTOMER-SPECIFIC rules that override general IFS knowledge.
- If a lookup fails, try alternative approaches before giving up.
- Order numbers with asterisks like "*1063" should include the asterisk.

## REASONING PROCESS (ReAct)
For multi-step tasks, follow this Think-Act-Observe cycle:

1. **THINK**: Before each action, state what you know and what you need to do next
2. **ACT**: Call exactly one tool
3. **OBSERVE**: Analyze the tool result - what did you learn? What changed?
4. **REPEAT**: Ask yourself "Is the task complete?" If not, continue the cycle

CRITICAL: After each tool call, you MUST think about whether more steps remain.
Do NOT stop after a single tool call unless the task is truly complete.

Example for "Move inventory via Shipment Order":
- THINK: I need to create a shipment order, then add a line, then release it. Step 1 is create.
- ACT: create_shipment_order(...)
- OBSERVE: Got shipment_order_id=34. This is step 1 of 3. I still need to add lines.
- THINK: Now I need to add the line item with the part and quantity.
- ACT: add_shipment_order_line(shipment_order_id=34, ...)
- OBSERVE: Line added. This is step 2 of 3. I still need to release.
- THINK: Finally, I need to release the shipment order.
- ACT: release_shipment_order(shipment_order_id=34)
- OBSERVE: Released. All 3 steps complete. Task is done.

## DATA INTEGRITY (CRITICAL)
- Never invent warehouses, locations, quantities, or sites.
- Only report values that appear in tool results.
- For write actions, never claim success unless tool result shows ok:true.
- When reporting IDs, use ONLY the value returned by the tool.

## TOOL CALL STRATEGY

### Independent Operations (CAN batch in parallel)
When calls don't depend on each other, you MAY batch them for efficiency:
- Checking inventory for multiple parts
- Looking up multiple orders
- Any read-only operations on different entities

### Dependent Workflows (MUST execute sequentially)
When one call's output is needed as input for the next, execute ONE AT A TIME:

Example - Shipment Order (SEQUENTIAL - each step needs previous result):
1. create_shipment_order → returns shipment_order_id (e.g., 34)
2. add_shipment_order_line → uses shipment_order_id from step 1
3. release_shipment_order → uses same shipment_order_id

CRITICAL: Never use placeholder values (like 0 or "TBD"). Wait for the actual response.

### Rules
- shipment_order_id is INTEGER (e.g., 34), never string like 'SO-34'
- Parameter is 'part_no' (not 'part' or 'part_number')
- If user doesn't specify part_no, ASK before creating the shipment header.
"""


async def process_message_streaming(user_message: str, event_queue: queue.Queue):
    """Process a user message with streaming output."""
    global _conversation_history

    tool_manager, client, args = await initialize()

    # Trim conversation history to prevent token overflow (keep last 10 exchanges)
    MAX_HISTORY_MESSAGES = 20  # 10 user + 10 assistant messages
    if len(_conversation_history) > MAX_HISTORY_MESSAGES:
        _conversation_history = _conversation_history[-MAX_HISTORY_MESSAGES:]

    # Smart tool filtering - get tools relevant to user intent
    relevant_tools = get_tools_for_intent(user_message)
    if relevant_tools:
        # Use filtered tool names (top 10 most relevant)
        tool_names = [t.name for t in relevant_tools]
    else:
        # Fallback to all tools if no matches
        tool_names = list(TOOL_REGISTRY.keys())

    # Retrieve relevant memories from past interactions
    memory_context = ""
    if tool_manager.memory_manager is not None:
        memory_context = tool_manager.get_relevant_memories_for_prompt(
            task_description=user_message,
            available_tool_names=tool_names,
            dataset_name='mcp',
        )

    # Build system prompt with memories
    system_with_memory = SYSTEM_PROMPT
    if memory_context:
        system_with_memory = f"{SYSTEM_PROMPT}\n\n{memory_context}"

    # Build messages
    messages = [{"role": "system", "content": system_with_memory}]
    messages.extend(_conversation_history)
    messages.append({"role": "user", "content": user_message})

    # Add user message to history
    _conversation_history.append({"role": "user", "content": user_message})

    max_iterations = 10
    iteration = 0
    full_response_parts = []

    # Memory folding tracking
    fold_count = 0
    accumulated_reasoning = ""
    tool_call_history_for_fold: List[Dict] = []

    user_lower = user_message.lower()

    mutating_request = any(
        word in user_lower
        for word in [
            'create', 'add', 'move', 'ship', 'shipment', 'reserve', 'release',
            'update', 'delete', 'cancel', 'cancelled', 'canceled'
        ]
    )

    def _looks_like_unverified_mutation_claim(text: str) -> bool:
        t = (text or '').lower()
        return any(
            k in t
            for k in [
                'created', 'released', 'updated', 'cancelled', 'canceled',
                'reserved', 'moved', 'shipment order', 'so-'
            ]
        )

    # Per-user-message guards to prevent repeated tool-call loops.
    executed_tool_signatures: set[str] = set()
    inventory_results_by_site: dict[str, dict] = {}
    inventory_part_no: str | None = None
    requested_sites = _extract_site_codes(user_message)
    wants_side_by_side = ("side-by-side" in user_lower) or ("compare" in user_lower)
    final_answer_emitted = False
    duplicate_only_rounds = 0

    shipment_ctx: dict[str, object] = {
        'shipment_order_id': None,
        'site': None,
        'from_warehouse': None,
        'to_warehouse': None,
        'line_no': None,
        'part_no': None,
        'qty_to_ship': None,
        'status': None,
    }
    shipment_flow_active = False

    # Task completion tracking (Plan/Track/Recover)
    workflow_info = detect_workflow(user_message)
    expected_tools: Optional[List[str]] = None
    tools_called_this_session: List[str] = []
    tool_search_done_this_session = False  # Track if LLM did a tool search (to get knowledge)

    if workflow_info and mutating_request:
        workflow_name, expected_tools = workflow_info
        event_queue.put({"type": "thinking", "step": 0, "status": f"Planning {workflow_name} workflow..."})

        # Generate task plan with next_actions checklist
        task_plan = await generate_task_plan(user_message)
        if task_plan:
            # Inject the plan into the system prompt for this request
            plan_text = f"\n\n## TASK PLAN (Generated)\n"
            if task_plan.get('immediate_goal'):
                plan_text += f"Goal: {task_plan['immediate_goal']}\n"
            if task_plan.get('next_actions'):
                plan_text += "Steps:\n"
                for i, action in enumerate(task_plan['next_actions'], 1):
                    desc = action.get('description', str(action))
                    plan_text += f"  {i}. {desc}\n"
            plan_text += "\nFollow this plan. Do NOT stop until all steps are complete.\n"

            # Inject into messages
            messages[0]["content"] = system_with_memory + plan_text

    while iteration < max_iterations:
        iteration += 1

        # Send thinking status
        event_queue.put({"type": "thinking", "step": iteration, "status": "Reasoning..."})

        # Stream LLM response
        try:
            stream = await client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                max_completion_tokens=4096,
                timeout=120,
                stream=True,
            )

            assistant_response = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    assistant_response += token
                    # Stream raw thinking tokens only when explicitly enabled.
                    if SHOW_RAW_THOUGHTS:
                        event_queue.put({"type": "thinking_token", "token": token})

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            event_queue.put({"type": "error", "message": error_msg})
            _conversation_history.append({"role": "assistant", "content": error_msg})
            event_queue.put({"type": "done"})
            return

        # Accumulate reasoning for memory folding
        accumulated_reasoning += assistant_response + "\n"

        # Extract and emit Chain of Thought for Activity Panel
        # This captures LLM reasoning that appears before tool calls
        cot_items = extract_chain_of_thought(assistant_response)
        for cot in cot_items:
            event_queue.put({
                "type": "cot",
                "cot_type": cot["type"],  # "reasoning"
                "content": cot["content"]
            })

        # Check for memory folding trigger (iteration threshold)
        if iteration >= FOLD_ITERATION_THRESHOLD and fold_count < MAX_FOLDS_PER_REQUEST:
            event_queue.put({"type": "thinking", "step": iteration, "status": "Folding memories..."})

            try:
                episode_mem, working_mem, tool_mem = await generate_memory_fold(
                    user_message=user_message,
                    reasoning_history=accumulated_reasoning,
                    tool_call_history=tool_call_history_for_fold,
                )

                # Store memories for cross-task learning
                if tool_manager.memory_manager is not None:
                    tool_manager.store_memory_on_fold(
                        episode_memory=episode_mem,
                        working_memory=working_mem,
                        tool_memory=tool_mem,
                        task_description=user_message,
                        dataset_name='mcp',
                    )

                # Inject memories into conversation to compress context
                memory_injection = f"""Memory of previous reasoning:

Episode Memory:
{json.dumps(episode_mem, indent=2)}

Working Memory:
{json.dumps(working_mem, indent=2)}

Tool Memory:
{json.dumps(tool_mem, indent=2)}

Continue reasoning from this compressed memory state."""

                messages.append({"role": "user", "content": memory_injection})
                accumulated_reasoning = ""  # Reset for next potential fold
                fold_count += 1
                print(f"  🧠 Memory fold #{fold_count} completed at iteration {iteration}")

            except Exception as e:
                print(f"  ⚠️ Memory fold failed: {e}")

        # Check for tool search request
        tool_search_query = extract_between(assistant_response, BEGIN_TOOL_SEARCH, END_TOOL_SEARCH)
        if tool_search_query and END_TOOL_SEARCH in assistant_response:
            # Handle tool search request
            event_queue.put({"type": "thinking", "step": iteration, "status": f"Searching tools: {tool_search_query[:50]}..."})

            try:
                # Initialize retriever (singleton - loads model once)
                retriever = MCPToolRetriever.get_instance()

                # Semantic search for relevant tools
                relevant_tool_names = retriever.retrieve(tool_search_query, top_k=10)

                # Get full schemas from MCP cache
                if hasattr(tool_manager, 'mcp_caller') and tool_manager.mcp_caller:
                    full_schemas = get_full_tool_schemas(relevant_tool_names, tool_manager.mcp_caller)
                else:
                    # Fallback - just return tool summaries
                    full_schemas = [
                        {"name": name, "summary": TOOL_REGISTRY[name].summary}
                        for name in relevant_tool_names if name in TOOL_REGISTRY
                    ]

                # Retrieve relevant knowledge for these tools
                # Pass mcp_caller to enable schema auto-extraction (Part 7)
                knowledge = {'procedural_rules': [], 'semantic_facts': [], 'error_corrections': []}
                if tool_manager.memory_manager:
                    mcp_caller = getattr(tool_manager, 'mcp_caller', None)
                    knowledge = tool_manager.memory_manager.retrieve_relevant_knowledge(
                        query=tool_search_query,
                        tool_names=relevant_tool_names,
                        mcp_caller=mcp_caller,
                    )

                # Format search result
                search_result = {
                    "tools": full_schemas,
                    "knowledge": {
                        "rules": knowledge.get('procedural_rules', []),
                        "facts": knowledge.get('semantic_facts', []),
                        "avoid_errors": knowledge.get('error_corrections', []),
                    }
                }

                # Inject into conversation
                result_text = f"\n\n{BEGIN_TOOL_SEARCH_RESULT}\n{json.dumps(search_result, indent=2)}\n{END_TOOL_SEARCH_RESULT}\n\n"

                messages.append({"role": "assistant", "content": assistant_response})
                messages.append({"role": "user", "content": result_text})

                event_queue.put({
                    "type": "tool_search",
                    "query": tool_search_query,
                    "tools_found": len(full_schemas),
                })

                print(f"  🔍 Tool search: '{tool_search_query}' → {len(full_schemas)} tools", flush=True)
                print(f"      Knowledge injected: {len(knowledge.get('procedural_rules', []))} rules, {len(knowledge.get('semantic_facts', []))} facts", flush=True)
                # Log first few rules for debugging
                for rule in knowledge.get('procedural_rules', [])[:3]:
                    print(f"      → {rule[:80]}...", flush=True)
                sys.stdout.flush()
                tool_search_done_this_session = True  # Mark that knowledge has been retrieved
                continue  # Get next LLM response with the tool schemas

            except Exception as e:
                print(f"  ⚠️ Tool search failed: {e}")
                # Continue without tool search results
                messages.append({"role": "assistant", "content": assistant_response})
                messages.append({"role": "user", "content": f"Tool search failed: {str(e)}. Please try calling tools directly using the available tools list."})
                continue

        # Check for fold_thought marker
        if FOLD_THOUGHT in assistant_response:
            event_queue.put({"type": "thinking", "step": iteration, "status": "Folding thoughts..."})

            try:
                episode_mem, working_mem, tool_mem = await generate_memory_fold(
                    user_message=user_message,
                    reasoning_history=accumulated_reasoning,
                    tool_call_history=tool_call_history_for_fold,
                )

                # Store memories
                if tool_manager.memory_manager:
                    tool_manager.memory_manager.store_complete_memory(
                        episode_memory=episode_mem,
                        working_memory=working_mem,
                        tool_memory=tool_mem,
                        task_description=user_message,
                        dataset_name='mcp',
                    )

                # Create compressed context
                fold_summary = f"""Previous reasoning compressed.
Progress: {json.dumps(episode_mem.get('current_progress', 'Unknown'))}
Key findings: {json.dumps(episode_mem.get('key_events', [])[:3])}
Tools used: {list(tool_mem.keys()) if isinstance(tool_mem, dict) else []}
Continue from this state."""

                # Reset conversation with fold summary
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": f"Let me continue from my previous progress.\n\n{fold_summary}"},
                ]
                accumulated_reasoning = ""
                fold_count += 1
                print(f"  📝 Explicit fold_thought triggered, fold #{fold_count}")
                continue

            except Exception as e:
                print(f"  ⚠️ Explicit fold_thought failed: {e}")
                # Continue without folding

        # Check for tool call(s)
        # ENFORCE: If LLM tries to call shipment tools without doing a tool search first,
        # force it to search to get the critical knowledge (e.g., 105 → AC, not AC-A105)
        if has_tool_call_marker(assistant_response) and not tool_search_done_this_session:
            # Check if it's trying to call a shipment-related tool
            shipment_tools = ['create_shipment_order', 'add_shipment_order_line', 'release_shipment_order']
            calling_shipment_tool = any(tool in assistant_response for tool in shipment_tools)

            if calling_shipment_tool and workflow_info:
                print(f"  🚫 ENFORCING tool search before shipment tools - LLM skipped it!", flush=True)
                messages.append({"role": "assistant", "content": assistant_response})
                messages.append({
                    "role": "user",
                    "content": (
                        "STOP. Before calling shipment tools, you MUST first use <tool_search> to retrieve "
                        "critical knowledge about warehouse IDs and parameters.\n\n"
                        "The knowledge base contains rules like:\n"
                        "- '105' refers to site AC itself (use from_warehouse='AC'), NOT 'AC-A105'\n"
                        "- '205' refers to warehouse AC-A205\n\n"
                        "Use <tool_search>create shipment order warehouse</tool_search> NOW to get the full rules."
                    )
                })
                continue
            elif calling_shipment_tool:
                print(f"  ⚠️ DIAGNOSTIC: LLM called shipment tool WITHOUT searching first!")

        if END_TOOL_CALL not in assistant_response:
            # If the model referenced tools but didn't use the required markers,
            # don't accept the response (it often leads to fabricated data).
            referenced_tools = [name for name in TOOL_REGISTRY.keys() if name in assistant_response]

            # Heuristic: inventory/stock questions must be backed by a tool call.
            inventory_query = (
                ("inventory" in user_lower or "stock" in user_lower)
                and ("part" in user_lower or "catalog" in user_lower)
                and ("site" in user_lower or "contract" in user_lower)
            )

            mutation_claim = mutating_request and _looks_like_unverified_mutation_claim(assistant_response)

            if referenced_tools or inventory_query or mutation_claim:
                messages.append({"role": "assistant", "content": assistant_response})
                messages.append({
                    "role": "user",
                    "content": (
                        "You must not claim write actions without tool execution. "
                        "Re-output ONLY the tool call block(s) using the exact markers, with valid JSON. "
                        "No prose.\n\n"
                        f"Tools referenced: {', '.join(referenced_tools[:10]) if referenced_tools else '(none)'}"
                    )
                })
                continue

            # No tool calls in response - check if task is actually complete
            if expected_tools and ENABLE_COMPLETION_CHECK:
                is_complete, recovery_summary = await check_task_completion(
                    user_message=user_message,
                    reasoning_history=accumulated_reasoning,
                    tool_history=tools_called_this_session,
                    expected_tools=expected_tools,
                )
                if not is_complete:
                    # Task is incomplete - inject recovery prompt and continue
                    event_queue.put({"type": "thinking", "step": iteration, "status": "Task incomplete, prompting continuation..."})
                    messages.append({"role": "assistant", "content": assistant_response})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"STOP - the task is NOT complete.\n\n"
                            f"Progress summary:\n{recovery_summary}\n\n"
                            f"You have only called: {', '.join(tools_called_this_session) or 'no tools yet'}.\n"
                            f"Expected workflow requires: {', '.join(expected_tools)}.\n\n"
                            f"Continue with the NEXT step. Do NOT provide a final answer until ALL steps are complete."
                        )
                    })
                    continue

            # Task is complete (or no workflow tracking) - send final response
            full_response_parts.append(assistant_response)
            event_queue.put({"type": "response", "content": assistant_response})
            final_answer_emitted = True
            break

        tool_call_jsons = extract_tool_calls_robust(assistant_response)
        if not tool_call_jsons:
            # Marker present but couldn't parse blocks - check completion first
            if expected_tools and ENABLE_COMPLETION_CHECK:
                is_complete, recovery_summary = await check_task_completion(
                    user_message=user_message,
                    reasoning_history=accumulated_reasoning,
                    tool_history=tools_called_this_session,
                    expected_tools=expected_tools,
                )
                if not is_complete:
                    messages.append({"role": "assistant", "content": assistant_response})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Task incomplete. Missing steps from workflow.\n"
                            f"Continue with the remaining tools: {', '.join(set(expected_tools) - set(tools_called_this_session))}"
                        )
                    })
                    continue

            # Treat as final response
            full_response_parts.append(assistant_response)
            event_queue.put({"type": "response", "content": assistant_response})
            final_answer_emitted = True
            break

        # Add the assistant response once, then append each tool response.
        messages.append({"role": "assistant", "content": assistant_response})

        executed_any_tool = False
        saw_any_tool_block = False

        for tool_call_json in tool_call_jsons:
            try:
                tool_call = json.loads(tool_call_json)
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments", {})

                # Validate tool call structure early; never execute unknown/blank tools.
                if not isinstance(tool_name, str) or not tool_name or tool_name not in TOOL_REGISTRY:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your tool call JSON is invalid: it must include a valid 'name' from the available tools list. "
                            "Re-output ONLY valid tool call block(s) (no prose), using the exact markers and valid JSON."
                        )
                    })
                    saw_any_tool_block = True
                    executed_any_tool = False
                    break
                if tool_args is None:
                    tool_args = {}
                if not isinstance(tool_args, dict):
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your tool call JSON is invalid: 'arguments' must be an object/dict. "
                            "Re-output ONLY valid tool call block(s) (no prose), using the exact markers and valid JSON."
                        )
                    })
                    saw_any_tool_block = True
                    executed_any_tool = False
                    break

                saw_any_tool_block = True
                signature = json.dumps({"name": tool_name, "arguments": tool_args}, sort_keys=True)
                if signature in executed_tool_signatures:
                    continue
                executed_tool_signatures.add(signature)

                # Send tool call event
                event_queue.put({
                    "type": "tool_call",
                    "name": tool_name,
                    "arguments": tool_args,
                    "step": iteration
                })

                adapted_call = {"function": {"name": tool_name, "arguments": tool_args}}

                event_queue.put({"type": "tool_executing", "name": tool_name})
                result = await tool_manager.call_tool(adapted_call, {})

                # Track shipment workflow state for deterministic summaries.
                if tool_name == "create_shipment_order" and isinstance(result, dict):
                    shipment_flow_active = True
                    shipment_ctx["site"] = tool_args.get("site") or shipment_ctx["site"]
                    shipment_ctx["from_warehouse"] = tool_args.get("from_warehouse") or shipment_ctx["from_warehouse"]
                    shipment_ctx["to_warehouse"] = tool_args.get("to_warehouse") or shipment_ctx["to_warehouse"]
                    if result.get("ok") is True:
                        shipment_ctx["shipment_order_id"] = result.get("shipment_order_id")

                if tool_name == "add_shipment_order_line" and isinstance(result, dict):
                    if result.get("ok") is True:
                        shipment_ctx["shipment_order_id"] = result.get("shipment_order_id") or shipment_ctx.get("shipment_order_id")
                        shipment_ctx["line_no"] = result.get("line_no")
                        shipment_ctx["part_no"] = result.get("part_no") or tool_args.get("part_no") or shipment_ctx.get("part_no")
                        shipment_ctx["qty_to_ship"] = result.get("qty_to_ship") or tool_args.get("qty_to_ship") or shipment_ctx.get("qty_to_ship")

                if tool_name == "release_shipment_order" and isinstance(result, dict):
                    if result.get("ok") is True:
                        shipment_ctx["shipment_order_id"] = result.get("shipment_order_id") or shipment_ctx.get("shipment_order_id")
                        shipment_ctx["status"] = result.get("status") or "Released"

                if tool_name == "get_inventory_stock":
                    site = tool_args.get("site")
                    if isinstance(site, str) and site:
                        inventory_results_by_site[site] = result if isinstance(result, dict) else {"ok": False, "error": str(result)}
                    if inventory_part_no is None:
                        part_no = tool_args.get("part_no")
                        if isinstance(part_no, str) and part_no:
                            inventory_part_no = part_no
                result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)

                # Truncate for display (shown to user)
                display_result = result_str[:3000] + "..." if len(result_str) > 3000 else result_str

                # Truncate for LLM context (prevents token overflow)
                MAX_RESULT_TOKENS = 8000  # ~2000 tokens worth of JSON
                if len(result_str) > MAX_RESULT_TOKENS:
                    truncated_result = result_str[:MAX_RESULT_TOKENS]
                    last_brace = truncated_result.rfind('},')
                    if last_brace > MAX_RESULT_TOKENS // 2:
                        truncated_result = truncated_result[:last_brace + 1]
                    record_count = result_str.count('"OrderNo"')
                    truncated_result += f'\n  // ... truncated, approximately {record_count} total records. Summarize what you can see.'
                    llm_result = truncated_result
                else:
                    llm_result = result_str

                event_queue.put({
                    "type": "tool_result",
                    "name": tool_name,
                    "result": display_result,
                    "success": True
                })

                tool_display = f"🔧 **Calling {tool_name}**\n```json\n{json.dumps(tool_args, indent=2)}\n```"
                full_response_parts.append(f"{tool_display}\n\n**Result:**\n```json\n{display_result}\n```")

                # Provide tool result back to the LLM for the next iteration.
                messages.append({
                    "role": "user",
                    "content": f"{BEGIN_TOOL_RESPONSE}\nTOOL: {tool_name}\n{llm_result}\n{END_TOOL_RESPONSE}"
                })

                # Track tool call for memory folding
                tool_call_history_for_fold.append({
                    "tool_call": {"name": tool_name, "arguments": tool_args},
                    "tool_response": result if isinstance(result, dict) else str(result)
                })

                # Track tools called for completion checking
                tools_called_this_session.append(tool_name)

                # Extract tool intent for progress tracking (if enabled)
                if ENABLE_INTENT_TRACKING and expected_tools:
                    intent = await extract_tool_intent(accumulated_reasoning)
                    if intent:
                        # Inject intent as a reminder to the LLM
                        messages.append({
                            "role": "user",
                            "content": f"<progress_note>{intent}</progress_note>"
                        })

                executed_any_tool = True

            except json.JSONDecodeError:
                # If one tool block is malformed, stop tool execution and return what we have.
                full_response_parts.append(assistant_response)
                event_queue.put({"type": "response", "content": assistant_response})
                final_answer_emitted = True
                break
            except Exception as e:
                error_msg = str(e)
                event_queue.put({
                    "type": "tool_result",
                    "name": tool_name if 'tool_name' in locals() else 'unknown',
                    "result": error_msg,
                    "success": False
                })
                tool_display = f"🔧 **Calling {tool_name}**\n```json\n{json.dumps(tool_args, indent=2)}\n```" if 'tool_name' in locals() else "🔧 **Calling unknown**"
                full_response_parts.append(f"{tool_display}\n\n❌ **Tool Error:** {error_msg}")
                break

        # If we executed any tools this turn, check if we need more tool calls or can finalize.
        if executed_any_tool:
            # Check if there are more sites to query (for multi-site inventory requests)
            pending_sites = [s for s in requested_sites if s not in inventory_results_by_site]

            # Only auto-continue site checks when we're actively doing inventory-by-site work.
            if pending_sites and inventory_results_by_site and iteration < max_iterations - 1:
                # Still have sites to check - prompt model to continue with remaining sites
                messages.append({
                    "role": "user",
                    "content": (
                        f"Good. Now continue checking the remaining sites: {', '.join(pending_sites)}. "
                        f"Call get_inventory_stock for each remaining site."
                    )
                })
                continue  # Continue loop to get more tool calls

            # If this is a side-by-side inventory comparison and we have results for all requested sites,
            # emit a deterministic summary to avoid hallucinations/loops.
            if wants_side_by_side and inventory_results_by_site and requested_sites:
                if all(site in inventory_results_by_site for site in requested_sites):
                    summary = _summarize_inventory_by_site({s: inventory_results_by_site[s] for s in requested_sites})
                    part_label = inventory_part_no or "(unknown part)"
                    final_text = (
                        f"Inventory comparison for part {part_label} (by site):\n\n{summary}\n\n"
                        "(Numbers are computed from tool results; sites with 0 locations have no stock returned.)"
                    )
                    full_response_parts.append(final_text)
                    event_queue.put({"type": "response", "content": final_text})
                    final_answer_emitted = True
                    break

            # Deterministic shipment summary (prevents hallucinated shipment order IDs).
            if shipment_flow_active and shipment_ctx.get("shipment_order_id"):
                so_id = shipment_ctx.get("shipment_order_id")
                site = shipment_ctx.get("site") or "(unknown site)"
                from_wh = shipment_ctx.get("from_warehouse") or "(unknown)"
                to_wh = shipment_ctx.get("to_warehouse") or "(unknown)"
                line_no = shipment_ctx.get("line_no")
                part_no = shipment_ctx.get("part_no") or "(unknown part)"
                qty = shipment_ctx.get("qty_to_ship")
                status = shipment_ctx.get("status") or "Planned"

                qty_text = _fmt_qty(qty) if qty is not None else "(unknown qty)"
                line_text = f"Line {line_no}" if line_no is not None else "Line"

                final_text = (
                    f"Created shipment order **{so_id}** at site **{site}** to move inventory **{from_wh} → {to_wh}**.\n\n"
                    f"- {line_text}: **{part_no}**, qty **{qty_text}**\n"
                    f"- Status: **{status}**"
                )
                full_response_parts.append(final_text)
                event_queue.put({"type": "response", "content": final_text})
                final_answer_emitted = True
                break

            # Request final summary from model
            messages.append({
                "role": "user",
                "content": (
                    "All requested tool calls have been executed and results are provided above. "
                    "Now produce a concise final answer summarizing the results. "
                    "Do NOT call any more tools - just summarize what was found."
                )
            })

            # Get final response
            try:
                final_stream = await client.chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    max_completion_tokens=2048,
                    timeout=60,
                    stream=True,
                )
                final_response = ""
                async for chunk in final_stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        final_response += token
                        if SHOW_RAW_THOUGHTS:
                            event_queue.put({"type": "thinking_token", "token": token})

                # Strip any tool call markers from final response (force text-only)
                if has_tool_call_marker(final_response):
                    # Model tried to call tools again - just use the text before the marker
                    # Handle both exact and malformed tags
                    final_response = re.split(r'<tool_call\b', final_response)[0].strip()
                    if not final_response:
                        final_response = "Tool results are shown above."

                full_response_parts.append(final_response)
                event_queue.put({"type": "response", "content": final_response})
                final_answer_emitted = True
            except Exception as e:
                event_queue.put({"type": "response", "content": f"Tool results shown above. (Summary error: {e})"})
                final_answer_emitted = True
            break

        # If the model only repeated already-executed tool calls, re-prompt for a final answer.
        if saw_any_tool_block and not executed_any_tool:
            duplicate_only_rounds += 1
            messages.append({
                "role": "user",
                "content": (
                    "Those tool calls were already executed earlier in this turn. "
                    "Do NOT repeat tool calls. Output ONLY the final answer now (no tool call markers)."
                )
            })
            if duplicate_only_rounds >= 2:
                final_text = (
                    "⚠️ The model kept repeating tool calls that were already executed. "
                    "Tool results are shown above; please ask for a specific summary format if needed."
                )
                full_response_parts.append(final_text)
                event_queue.put({"type": "response", "content": final_text})
                final_answer_emitted = True
                break

    if iteration >= max_iterations:
        full_response_parts.append("\n\n⚠️ *Reached maximum tool call limit*")
        event_queue.put({"type": "warning", "message": "Reached maximum tool call limit"})

    # Ensure non-streaming /chat gets a final response event even if we only emitted tool events.
    if not final_answer_emitted and full_response_parts:
        event_queue.put({"type": "response", "content": "\n\n---\n\n".join(full_response_parts[-1:])})

    # Build final response for history
    full_response = "\n\n---\n\n".join(full_response_parts)
    _conversation_history.append({"role": "assistant", "content": full_response})

    # Store memories for learning from this interaction
    if tool_manager.memory_manager is not None and executed_tool_signatures:
        try:
            # Build tool memory from executed tools
            tools_used = []
            for sig in executed_tool_signatures:
                sig_data = json.loads(sig)
                tool_name = sig_data.get("name", "unknown")
                tools_used.append({
                    "tool_name": tool_name,
                    "success_rate": 1.0,  # Completed without error
                    "effective_parameters": list(sig_data.get("arguments", {}).keys()),
                })

            tool_memory = {"tools_used": tools_used, "derived_rules": []}

            # Build episodic memory
            episode_memory = {
                "task_description": user_message[:200],
                "key_events": [{"outcome": f"Executed {len(executed_tool_signatures)} tool(s) successfully"}],
                "current_progress": "completed",
            }

            # Store (working memory is session-only so we pass empty dict)
            tool_manager.store_memory_on_fold(
                episode_memory=episode_memory,
                working_memory={},
                tool_memory=tool_memory,
                task_description=user_message,
                dataset_name='mcp',
            )
        except Exception:
            pass  # Don't fail the request if memory storage fails

    event_queue.put({"type": "done"})


def run_async_streaming(user_message: str, event_queue: queue.Queue):
    """Run the async streaming function in a new event loop per request.

    Each request gets its own event loop to avoid 'event loop already running' errors
    when multiple requests come in concurrently.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(process_message_streaming(user_message, event_queue))
    finally:
        loop.close()


# ============================================================================
# Native Function Calling Agent (v2)
# ============================================================================

_native_agent = None
_native_conversation_history = []


async def initialize_native_agent():
    """Initialize the native function calling agent."""
    global _native_agent, _tool_manager, _client, _args

    # Ensure base initialization is done
    await initialize()

    if _native_agent is None:
        from agents.native_tool_agent import NativeToolAgent
        from prompts.prompts_native import get_native_prompt

        # Get tool retriever from registry
        tool_retriever = None
        try:
            tool_retriever = MCPToolRetriever()
        except Exception as e:
            print(f"Warning: Could not create tool retriever: {e}")

        _native_agent = NativeToolAgent(
            client=_client,
            model_name=_args.model_name,
            tool_manager=_tool_manager.mcp_caller,
            tool_retriever=tool_retriever,
            memory_manager=_tool_manager.memory_manager,
            system_prompt=get_native_prompt(),
            max_iterations=30,
        )

    return _native_agent


async def process_message_native(user_message: str, event_queue: queue.Queue):
    """Process a message using the native function calling agent."""
    global _native_conversation_history

    try:
        agent = await initialize_native_agent()

        # Run agent and stream events
        async for event in agent.run(user_message, _native_conversation_history):
            event_dict = event.to_dict()

            # Map event types to existing frontend format
            if event.type == "thinking":
                event_queue.put({
                    "type": "raw_thought",
                    "content": event.data.get("content", "")
                })
            elif event.type == "tool_call":
                event_queue.put({
                    "type": "tool_call",
                    "name": event.data.get("name", ""),
                    "arguments": event.data.get("arguments", {})
                })
            elif event.type == "tool_result":
                result = event.data.get("result", {})
                success = "error" not in result if isinstance(result, dict) else True
                event_queue.put({
                    "type": "tool_result",
                    "name": event.data.get("name", ""),
                    "success": success,
                    "result": json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
                })
            elif event.type == "response":
                content = event.data.get("content", "")
                if content:
                    event_queue.put({
                        "type": "response",
                        "content": content
                    })
                    # Update conversation history
                    _native_conversation_history.append({"role": "user", "content": user_message})
                    _native_conversation_history.append({"role": "assistant", "content": content})
            elif event.type == "error":
                event_queue.put({
                    "type": "error",
                    "message": event.data.get("message", "Unknown error")
                })
            elif event.type == "done":
                pass  # Will send done below

    except Exception as e:
        import traceback
        traceback.print_exc()
        event_queue.put({
            "type": "error",
            "message": f"Agent error: {str(e)}"
        })

    event_queue.put({"type": "done"})


def run_async_native(user_message: str, event_queue: queue.Queue):
    """Run the native agent in a new event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(process_message_native(user_message, event_queue))
    finally:
        loop.close()


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepAgent</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-primary: #f9f5f1;
            --bg-secondary: #ffffff;
            --bg-tertiary: #f3ebe4;
            --text-primary: #1a1a1a;
            --text-secondary: #5c5c5c;
            --text-tertiary: #8b8b8b;
            --accent: #c96442;
            --accent-hover: #b5573a;
            --accent-light: #fff5f2;
            --border: #e5ddd5;
            --border-light: #ebe5df;
            --user-bg: #ebe5df;
            --assistant-bg: #ffffff;
            --success: #2e7d32;
            --success-bg: #e8f5e9;
            --error: #c62828;
            --error-bg: #ffebee;
            --pending: #c96442;
            --pending-bg: #fff5f2;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
            --shadow-md: 0 2px 8px rgba(0,0,0,0.06);
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 20px;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            font-size: 15px;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }

        /* Header */
        header {
            background: var(--bg-secondary);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: var(--shadow-sm);
        }
        .header-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .header-logo {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, var(--accent) 0%, #e07a5f 100%);
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 14px;
        }
        header h1 {
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        header p {
            color: var(--text-tertiary);
            font-size: 0.8rem;
            margin-top: 1px;
        }
        .header-controls { display: flex; gap: 0.5rem; align-items: center; }
        .toggle-btn {
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            padding: 0.5rem 0.875rem;
            border-radius: var(--radius-lg);
            font-size: 0.8rem;
            font-weight: 500;
            border: 1px solid var(--border);
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .toggle-btn:hover { background: var(--border-light); }
        .toggle-btn.active {
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }

        #main-container { flex: 1; display: flex; overflow: hidden; }

        /* Activity Panel */
        #thinking-panel {
            width: 0;
            overflow: hidden;
            transition: width 0.25s ease;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }
        #thinking-panel.visible { width: 260px; }
        #thinking-header {
            padding: 0.875rem 1rem;
            border-bottom: 1px solid var(--border-light);
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-tertiary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--text-tertiary);
            transition: background 0.2s;
        }
        .status-dot.active {
            background: var(--success);
            box-shadow: 0 0 0 3px var(--success-bg);
            animation: pulse 1.5s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(0.95); }
        }
        #thinking-content {
            flex: 1;
            overflow-y: auto;
            padding: 0.75rem;
        }

        /* Timeline */
        .timeline-item {
            display: flex;
            align-items: flex-start;
            gap: 0.625rem;
            padding: 0.5rem 0;
            position: relative;
        }
        .timeline-item:not(:last-child)::before {
            content: '';
            position: absolute;
            left: 9px;
            top: 26px;
            width: 1px;
            height: calc(100% - 10px);
            background: var(--border-light);
        }
        .timeline-icon {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.65rem;
            flex-shrink: 0;
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            font-weight: 500;
        }
        .timeline-icon.pending {
            background: var(--pending-bg);
            color: var(--pending);
        }
        .timeline-icon.success {
            background: var(--success-bg);
            color: var(--success);
        }
        .timeline-icon.error {
            background: var(--error-bg);
            color: var(--error);
        }
        .timeline-icon.thinking {
            background: #e8f4fd;
            color: #1976d2;
        }
        .timeline-icon.observing {
            background: #f3e8fd;
            color: #7b1fa2;
        }
        .timeline-icon.reasoning {
            background: #fff8e1;
            color: #f57c00;
        }
        .timeline-content { flex: 1; min-width: 0; }
        .timeline-title {
            font-size: 0.8rem;
            color: var(--text-primary);
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .timeline-subtitle {
            font-size: 0.7rem;
            color: var(--text-tertiary);
            margin-top: 1px;
        }
        .timeline-time {
            font-size: 0.65rem;
            color: var(--text-tertiary);
            margin-top: 2px;
            font-variant-numeric: tabular-nums;
        }

        /* Chat Panel */
        #chat-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
            background: var(--bg-primary);
        }
        #chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem 2rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            max-width: 800px;
            margin: 0 auto;
            width: 100%;
        }
        .message {
            padding: 1rem 1.25rem;
            border-radius: var(--radius-md);
            line-height: 1.65;
            box-shadow: var(--shadow-sm);
        }
        .user {
            background: var(--user-bg);
            align-self: flex-end;
            max-width: 75%;
            border: 1px solid var(--border-light);
        }
        .assistant {
            background: var(--assistant-bg);
            align-self: flex-start;
            max-width: 85%;
            border: 1px solid var(--border);
        }
        .assistant.streaming {
            border-color: var(--accent);
            box-shadow: 0 0 0 2px var(--accent-light);
        }
        .assistant p { margin-bottom: 0.75rem; }
        .assistant p:last-child { margin-bottom: 0; }
        .assistant strong { font-weight: 600; }
        .assistant pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 1rem;
            border-radius: var(--radius-sm);
            overflow-x: auto;
            font-size: 0.8rem;
            margin: 0.75rem 0;
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
        }
        .assistant code {
            background: var(--bg-tertiary);
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.85em;
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
        }
        .assistant pre code {
            background: none;
            padding: 0;
        }
        .assistant ul, .assistant ol {
            margin: 0.5rem 0 0.5rem 1.25rem;
        }
        .assistant li { margin-bottom: 0.25rem; }

        /* Working indicator */
        .working-indicator {
            color: var(--text-secondary);
            font-style: normal;
            display: flex;
            align-items: center;
            gap: 0.625rem;
            font-size: 0.9rem;
        }
        .working-indicator::before {
            content: '';
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent);
            animation: pulse 1.5s ease-in-out infinite;
        }

        /* Input Area */
        #input-area {
            background: var(--bg-secondary);
            padding: 1rem 2rem 1.25rem;
            border-top: 1px solid var(--border);
        }
        #input-wrapper {
            max-width: 800px;
            margin: 0 auto;
        }
        #input-row {
            display: flex;
            gap: 0.75rem;
            align-items: flex-end;
        }
        #message-input {
            flex: 1;
            padding: 0.75rem 1rem;
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: 0.95rem;
            font-family: inherit;
            resize: none;
            transition: border-color 0.15s, box-shadow 0.15s;
            min-height: 42px;
            max-height: 200px;
            overflow-y: auto;
        }
        #message-input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-light);
        }
        #message-input::placeholder { color: var(--text-tertiary); }
        button {
            padding: 0.625rem 1.25rem;
            border: none;
            border-radius: var(--radius-sm);
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 500;
            font-family: inherit;
            transition: all 0.15s ease;
        }
        #send-btn {
            background: var(--accent);
            color: white;
        }
        #send-btn:hover { background: var(--accent-hover); }
        #send-btn:disabled {
            background: var(--border);
            color: var(--text-tertiary);
            cursor: not-allowed;
        }
        #clear-btn {
            background: transparent;
            color: var(--text-secondary);
            border: 1px solid var(--border);
        }
        #clear-btn:hover {
            background: var(--bg-tertiary);
        }

        /* Quick Actions */
        .quick-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            padding: 0.75rem 2rem;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-light);
        }
        .quick-action {
            background: var(--bg-primary);
            color: var(--text-secondary);
            padding: 0.5rem 0.875rem;
            border-radius: var(--radius-lg);
            font-size: 0.8rem;
            border: 1px solid var(--border);
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .quick-action:hover {
            background: var(--bg-tertiary);
            border-color: var(--border);
            color: var(--text-primary);
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-tertiary); }
    </style>
</head>
<body>
    <header>
        <div class="header-brand">
            <div class="header-logo">D</div>
            <div>
                <h1>DeepAgent</h1>
                <p>IFS Cloud Assistant</p>
            </div>
        </div>
        <div class="header-controls">
            <button class="toggle-btn" id="agent-toggle" onclick="toggleAgent()">
                v2 Agent
            </button>
            <button class="toggle-btn active" id="thinking-toggle" onclick="toggleThinking()">
                Activity
            </button>
        </div>
    </header>

    <div class="quick-actions">
        <button class="quick-action" onclick="sendExample('Check if the MCP connection is working')">Test Connection</button>
        <button class="quick-action" onclick="sendExample('Show me any past due customer order lines')">Past Due Orders</button>
        <button class="quick-action" onclick="sendExample('What is the inventory for part 10106105 at site AC?')">Check Inventory</button>
        <button class="quick-action" onclick="sendExample('Search for customers with Costco in the name')">Find Customers</button>
    </div>

    <div id="main-container">
        <div id="thinking-panel" class="visible">
            <div id="thinking-header">
                <span class="status-dot" id="thinking-status"></span>
                Activity
            </div>
            <div id="thinking-content"></div>
        </div>

        <div id="chat-panel">
            <div id="chat-container"></div>
            <div id="input-area">
                <div id="input-wrapper">
                    <div id="input-row">
                        <textarea id="message-input" rows="1" placeholder="Message DeepAgent..."></textarea>
                        <button id="send-btn" onclick="sendMessage()">Send</button>
                        <button id="clear-btn" onclick="clearChat()">Clear</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const showRawThoughts = {{ 'true' if show_raw_thoughts else 'false' }};

        const chatContainer = document.getElementById('chat-container');
        const thinkingContent = document.getElementById('thinking-content');
        const thinkingStatus = document.getElementById('thinking-status');
        const thinkingPanel = document.getElementById('thinking-panel');
        const thinkingToggle = document.getElementById('thinking-toggle');
        const messageInput = document.getElementById('message-input');
        const sendBtn = document.getElementById('send-btn');

        let currentAssistantMessage = null;
        let currentTimelineItem = null;
        let initialTimelineItem = null;
        let toolCallCount = 0;
        let startTime = null;

        let rawThoughtsEl = null;
        let rawThoughtsBuffer = '';
        let rawThoughtsDirty = false;

        // Agent mode toggle (v1 = XML tags, v2 = native function calling)
        let useNativeAgent = false;
        const agentToggle = document.getElementById('agent-toggle');

        function toggleAgent() {
            useNativeAgent = !useNativeAgent;
            agentToggle.classList.toggle('active', useNativeAgent);
            agentToggle.textContent = useNativeAgent ? 'v2 Agent (ON)' : 'v2 Agent';
        }

        function toggleThinking() {
            thinkingPanel.classList.toggle('visible');
            thinkingToggle.classList.toggle('active');
        }

        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Auto-resize textarea
        messageInput.addEventListener('input', () => {
            messageInput.style.height = 'auto';
            messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
        });

        function addUserMessage(content) {
            const div = document.createElement('div');
            div.className = 'message user';
            div.textContent = content;
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function createAssistantMessage() {
            const div = document.createElement('div');
            div.className = 'message assistant streaming';
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return div;
        }

        function formatToolName(name) {
            return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        }

        function getElapsedTime() {
            if (!startTime) return '';
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            return `${elapsed}s`;
        }

        function addTimelineItem(icon, iconClass, title, subtitle) {
            const item = document.createElement('div');
            item.className = 'timeline-item';
            item.innerHTML = `
                <div class="timeline-icon ${iconClass}">${icon}</div>
                <div class="timeline-content">
                    <div class="timeline-title">${title}</div>
                    ${subtitle ? `<div class="timeline-subtitle">${subtitle}</div>` : ''}
                    <div class="timeline-time">${getElapsedTime()}</div>
                </div>
            `;
            thinkingContent.appendChild(item);
            thinkingContent.scrollTop = thinkingContent.scrollHeight;
            return item;
        }

        function updateTimelineText(item, title, subtitle) {
            if (!item) return;
            const titleEl = item.querySelector('.timeline-title');
            if (titleEl && typeof title === 'string') titleEl.textContent = title;

            const contentEl = item.querySelector('.timeline-content');
            if (!contentEl) return;

            let subtitleEl = item.querySelector('.timeline-subtitle');
            if (subtitle) {
                if (!subtitleEl) {
                    subtitleEl = document.createElement('div');
                    subtitleEl.className = 'timeline-subtitle';
                    const timeEl = item.querySelector('.timeline-time');
                    contentEl.insertBefore(subtitleEl, timeEl || null);
                }
                subtitleEl.textContent = subtitle;
            } else if (subtitleEl) {
                subtitleEl.remove();
            }
        }

        function updateTimelineItem(item, icon, iconClass) {
            if (item) {
                const iconEl = item.querySelector('.timeline-icon');
                iconEl.className = `timeline-icon ${iconClass}`;
                iconEl.textContent = icon;
                const timeEl = item.querySelector('.timeline-time');
                timeEl.textContent = getElapsedTime();
            }
        }

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            addUserMessage(message);
            messageInput.value = '';
            sendBtn.disabled = true;
            sendBtn.textContent = 'Working...';

            // Clear thinking panel for new message
            thinkingContent.innerHTML = '';
            toolCallCount = 0;
            startTime = Date.now();
            currentTimelineItem = null;
            initialTimelineItem = null;
            thinkingStatus.classList.add('active');

            rawThoughtsEl = null;
            rawThoughtsBuffer = '';
            rawThoughtsDirty = false;

            // Add initial timeline item
            initialTimelineItem = addTimelineItem('⏳', 'pending', 'Processing request...', '');

            // Create streaming assistant message
            currentAssistantMessage = createAssistantMessage();
            currentAssistantMessage.innerHTML = '<span class="working-indicator">Working...</span>';
            let finalResponse = '';

            try {
                const endpoint = useNativeAgent ? '/chat/stream/v2' : '/chat/stream';
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const event = JSON.parse(line.slice(6));
                                finalResponse = handleStreamEvent(event, finalResponse);
                            } catch (e) {
                                console.error('Parse error:', e);
                            }
                        }
                    }
                }
            } catch (error) {
                currentAssistantMessage.innerHTML = marked.parse('❌ Error: ' + error.message);
            }

            // Finalize
            thinkingStatus.classList.remove('active');
            currentAssistantMessage.classList.remove('streaming');
            sendBtn.disabled = false;
            sendBtn.textContent = 'Send';
            messageInput.focus();
        }

        function handleStreamEvent(event, finalResponse) {
            switch (event.type) {
                case 'thinking':
                    // Lightweight progress updates (no raw-thought leakage)
                    if (!initialTimelineItem) {
                        initialTimelineItem = addTimelineItem('⏳', 'pending', 'Processing request...', '');
                    }
                    if (event && (event.step || event.status)) {
                        const stepText = event.step ? `Step ${event.step}` : 'Step';
                        const statusText = event.status ? String(event.status) : 'Reasoning...';
                        updateTimelineText(initialTimelineItem, 'Processing request...', `${stepText}: ${statusText}`);
                        const timeEl = initialTimelineItem.querySelector('.timeline-time');
                        if (timeEl) timeEl.textContent = getElapsedTime();
                    }
                    break;

                case 'thinking_token':
                    if (!showRawThoughts) break;

                    if (!rawThoughtsEl) {
                        rawThoughtsEl = document.createElement('div');
                        rawThoughtsEl.className = 'timeline-item';
                        rawThoughtsEl.innerHTML = `
                            <div class="timeline-icon pending">🧠</div>
                            <div class="timeline-content">
                                <div class="timeline-title">Raw thoughts</div>
                                <pre class="timeline-subtitle" style="white-space: pre-wrap; margin: 6px 0 0 0;"></pre>
                                <div class="timeline-time">${getElapsedTime()}</div>
                            </div>
                        `;
                        thinkingContent.appendChild(rawThoughtsEl);
                        thinkingContent.scrollTop = thinkingContent.scrollHeight;
                    }

                    rawThoughtsBuffer += (event.token || '');

                    if (!rawThoughtsDirty) {
                        rawThoughtsDirty = true;
                        requestAnimationFrame(() => {
                            rawThoughtsDirty = false;
                            const pre = rawThoughtsEl.querySelector('pre');
                            if (pre) pre.textContent = rawThoughtsBuffer;
                            const timeEl = rawThoughtsEl.querySelector('.timeline-time');
                            if (timeEl) timeEl.textContent = getElapsedTime();
                            thinkingContent.scrollTop = thinkingContent.scrollHeight;
                        });
                    }
                    break;

                case 'cot':
                    // Chain of Thought - LLM reasoning extracted before tool calls
                    const cotType = event.cot_type;
                    const cotContent = event.content || '';
                    // Use lightbulb for reasoning (the LLM's thought process)
                    const cotIcon = '💭';
                    const cotTitle = 'Reasoning';
                    const cotClass = 'reasoning';
                    addTimelineItem(cotIcon, cotClass, cotTitle, cotContent);
                    break;

                case 'tool_call':
                    toolCallCount++;
                    const toolName = formatToolName(event.name);
                    // Get a brief summary of key args
                    const argKeys = Object.keys(event.arguments || {});
                    const argHint = argKeys.length > 0 ? argKeys.slice(0, 2).join(', ') : '';
                    currentTimelineItem = addTimelineItem('🔧', 'pending', toolName, argHint);
                    // Update chat with working indicator
                    updateWorkingIndicator(event.name);
                    break;

                case 'tool_executing':
                    break;

                case 'tool_result':
                    // Update the current timeline item with result status
                    if (event.success) {
                        updateTimelineItem(currentTimelineItem, '✓', 'success');
                    } else {
                        updateTimelineItem(currentTimelineItem, '✗', 'error');
                        finalResponse = `❌ **Tool Error:** ${event.result}`;
                        updateAssistantMessage(finalResponse);
                    }
                    break;

                case 'response':
                    // Add completion item and show response in chat
                    addTimelineItem('✓', 'success', 'Complete', `${toolCallCount} tool${toolCallCount !== 1 ? 's' : ''} called`);
                    finalResponse = event.content;
                    updateAssistantMessage(finalResponse);
                    break;

                case 'error':
                    addTimelineItem('✗', 'error', 'Error', event.message);
                    finalResponse = `❌ ${event.message}`;
                    updateAssistantMessage(finalResponse);
                    break;

                case 'warning':
                    addTimelineItem('⚠', 'pending', 'Warning', event.message);
                    break;

                case 'done':
                    // Streaming complete
                    break;
            }
            return finalResponse;
        }

        function updateWorkingIndicator(toolName) {
            if (currentAssistantMessage) {
                const toolLabel = formatToolName(toolName);
                currentAssistantMessage.innerHTML = `<span class="working-indicator">${toolLabel}...</span>`;
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }

        function updateAssistantMessage(content) {
            if (currentAssistantMessage && content) {
                currentAssistantMessage.innerHTML = marked.parse(content);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }

        function sendExample(text) {
            messageInput.value = text;
            sendMessage();
        }

        async function clearChat() {
            const clearEndpoint = useNativeAgent ? '/clear/v2' : '/clear';
            await fetch(clearEndpoint, { method: 'POST' });
            chatContainer.innerHTML = '';
            thinkingContent.innerHTML = '';
        }

        messageInput.focus();
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, show_raw_thoughts=SHOW_RAW_THOUGHTS)


@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint using Server-Sent Events."""
    data = request.json
    message = data.get('message', '')

    if not message:
        def empty_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Please enter a message.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return Response(empty_gen(), mimetype='text/event-stream')

    def generate():
        event_queue = queue.Queue()

        # Start async processing in a thread
        thread = threading.Thread(target=run_async_streaming, args=(message, event_queue))
        thread.start()

        # Stream events from queue
        while True:
            try:
                event = event_queue.get(timeout=120)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get('type') == 'done':
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Timeout waiting for response'})}\n\n"
                break

        thread.join(timeout=5)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/chat', methods=['POST'])
def chat():
    """Non-streaming fallback endpoint."""
    data = request.json
    message = data.get('message', '')

    if not message:
        return jsonify({'response': 'Please enter a message.'})

    event_queue = queue.Queue()
    run_async_streaming(message, event_queue)

    # Collect all events into a response
    response_parts = []
    while True:
        try:
            # Wait for the full run to complete; tools/LLM calls can legitimately take time.
            event = event_queue.get(timeout=120)
            if event.get('type') == 'done':
                break
            elif event.get('type') == 'response':
                response_parts.append(event.get('content', ''))
            elif event.get('type') == 'tool_call':
                args_str = json.dumps(event.get('arguments', {}), indent=2)
                response_parts.append(f"🔧 **Calling {event.get('name')}**\n```json\n{args_str}\n```")
            elif event.get('type') == 'tool_result':
                if event.get('success'):
                    response_parts.append(f"**Result:**\n```json\n{event.get('result', '')}\n```")
                else:
                    response_parts.append(f"❌ **Tool Error:** {event.get('result', '')}")
        except queue.Empty:
            response_parts.append("❌ **Error:** Timeout waiting for response")
            break

    return jsonify({'response': '\n\n---\n\n'.join(response_parts)})


@app.route('/clear', methods=['POST'])
def clear():
    global _conversation_history
    _conversation_history = []
    return jsonify({'status': 'ok'})


# ============================================================================
# Native Agent v2 Endpoints
# ============================================================================

@app.route('/chat/stream/v2', methods=['POST'])
def chat_stream_v2():
    """Streaming chat endpoint using native function calling agent."""
    data = request.json
    message = data.get('message', '')

    if not message:
        def empty_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Please enter a message.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return Response(empty_gen(), mimetype='text/event-stream')

    def generate():
        event_queue = queue.Queue()

        # Start native agent processing in a thread
        thread = threading.Thread(target=run_async_native, args=(message, event_queue))
        thread.start()

        # Stream events from queue
        while True:
            try:
                event = event_queue.get(timeout=120)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get('type') == 'done':
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Timeout waiting for response'})}\n\n"
                break

        thread.join(timeout=5)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/clear/v2', methods=['POST'])
def clear_v2():
    """Clear conversation history for native agent."""
    global _native_conversation_history, _native_agent
    _native_conversation_history = []
    # Reset agent tool state
    if _native_agent is not None:
        _native_agent.reset_tools()
    return jsonify({'status': 'ok'})


# ============================================================================
# Claude Code Style Agent (v3)
# ============================================================================

_v3_agent = None


@app.route('/chat/stream/v3', methods=['POST'])
def chat_stream_v3():
    """Claude Code style agent endpoint."""
    global _v3_agent

    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Lazy initialize v3 agent
    if _v3_agent is None:
        from agent import Agent
        try:
            _v3_agent = Agent.from_config("config/base_config.yaml")
        except Exception as e:
            return jsonify({'error': f'Failed to initialize agent: {e}'}), 500

    def generate():
        try:
            result = _v3_agent.run(user_message)
            # Send result as response event
            yield f"data: {json.dumps({'type': 'response', 'content': result})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/clear/v3', methods=['POST'])
def clear_v3():
    """Clear v3 agent state."""
    global _v3_agent
    _v3_agent = None
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("Starting DeepAgent + MCP Chat UI with Streaming...")
    print("Make sure MCP servers are running")

    host = os.getenv("DEEPAGENT_HOST", "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(os.getenv("DEEPAGENT_PORT", "7865"))
    except ValueError:
        port = 7865

    # If binding 0.0.0.0, still suggest 127.0.0.1 for local browsing.
    suggested_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"\n  🌐 Open http://{suggested_host}:{port} in your browser\n")
    app.run(host=host, port=port, debug=False, threaded=True)
