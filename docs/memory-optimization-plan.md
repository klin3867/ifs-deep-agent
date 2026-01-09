# Memory Optimization Plan for DeepAgent

## Context
This plan optimizes the DeepAgent memory system for IFS Cloud ERP integration. The codebase is at `/Users/jimmykline/dev/DeepAgent`.

**Problem:** The current system loads procedural rules as "memories" dynamically, sends all 59 tools to the LLM on every request, and lacks semantic domain knowledge retrieval.

**Solution:** Separate concerns into 4 memory types with appropriate storage and retrieval mechanisms.

## Target Architecture

| Memory Type | Purpose | Storage | Retrieval |
|-------------|---------|---------|-----------|
| Procedural | How to use tools correctly | Embedded in tool descriptions (mcp_tool_registry.py) | Always included with tool |
| Semantic | IFS domain facts (warehouses, sites) | config/ifs_semantic.yaml | Keyword matching |
| Episodic | Past task outcomes | cache/memory/episodic_memories.json | Similarity scoring |
| Working | Current session state | In-memory only | Always available |

---

## Implementation Steps

### Step 1: Enhance Tool Registry with Procedural Rules

**File:** `src/tools/mcp_tool_registry.py`

#### 1a. Add `rules` field to ToolSummary NamedTuple (around line 79):

```python
class ToolSummary(NamedTuple):
    """Compact tool metadata for LLM decision-making."""
    name: str
    server: str
    summary: str  # One line, <100 chars
    category: str  # orders, customers, inventory, shipments, planning, reservations
    mutates: bool  # True for create/update/delete
    use_when: str  # When would someone need this?
    rules: Tuple[str, ...] = ()  # Procedural rules for this tool
```

#### 1b. Update shipment tools with embedded rules (around line 438):

```python
"create_shipment_order": ToolSummary(
    name="create_shipment_order",
    server="planning",
    summary="Create a new shipment order",
    category="shipments",
    mutates=True,
    use_when="User wants to create a shipment, start new shipment order",
    rules=(
        "Always determine part_no BEFORE creating - ask user if not specified",
        "shipment_order_id returned is INTEGER (e.g., 34), not string like 'SO-123'",
        "Site AC remote warehouses: AC-A110, AC-A205. 'Warehouse 105' = site 'AC'",
    ),
),
"add_shipment_order_line": ToolSummary(
    name="add_shipment_order_line",
    server="planning",
    summary="Add a line item to a shipment order",
    category="shipments",
    mutates=True,
    use_when="User wants to add parts to a shipment",
    rules=(
        "Use EXACT shipment_order_id INTEGER from create_shipment_order response",
        "Required: shipment_order_id (int), part_no, qty_to_ship",
    ),
),
"release_shipment_order": ToolSummary(
    name="release_shipment_order",
    server="planning",
    summary="Release a shipment order for processing",
    category="shipments",
    mutates=True,
    use_when="User wants to release/approve shipment for warehouse",
    rules=(
        "Only release after successfully adding at least one line",
        "Use same integer shipment_order_id from create response",
    ),
),
```

#### 1c. Update `get_tool_summaries_for_prompt()` to include rules (around line 613):

```python
def get_tool_summaries_for_prompt(
    categories: Optional[List[str]] = None,
    exclude_categories: Optional[List[str]] = None,
    mutating_only: bool = False,
    read_only: bool = False,
    include_rules: bool = True,  # NEW parameter
) -> str:
    """
    Generate compact tool list for LLM prompt.

    Args:
        categories: Only include tools from these categories
        exclude_categories: Exclude tools from these categories
        mutating_only: Only include tools that mutate state
        read_only: Only include tools that do not mutate state
        include_rules: Include procedural rules for each tool

    Returns:
        Formatted string with tool summaries, ~20 tokens per tool
    """
    tools = list(TOOL_REGISTRY.values())

    if categories:
        tools = [t for t in tools if t.category in categories]
    if exclude_categories:
        tools = [t for t in tools if t.category not in exclude_categories]
    if mutating_only:
        tools = [t for t in tools if t.mutates]
    if read_only:
        tools = [t for t in tools if not t.mutates]

    lines = []
    for t in sorted(tools, key=lambda x: x.category + x.name):
        flag = " [WRITE]" if t.mutates else ""
        lines.append(f"- {t.name}: {t.summary}{flag}")
        if include_rules and t.rules:
            for rule in t.rules:
                lines.append(f"    • {rule}")

    return "\n".join(lines)
```

---

### Step 2: Create Semantic Memory Config

**File:** `config/ifs_semantic.yaml` (NEW FILE)

```yaml
# IFS Semantic Memory - Domain knowledge for DeepAgent

sites:
  keywords: [site, sites, contract, location, warehouse, warehouses]
  facts:
    - "Site AC has remote warehouses: AC-A110, AC-A205"
    - "Warehouse names with '105' (e.g., 'Warehouse 105') refer to site AC itself"
    - "Default site is 'AC' unless user specifies otherwise"

shipments:
  keywords: [shipment, ship, move, transfer, transport]
  facts:
    - "Shipment orders require 3 steps: create_shipment_order -> add_shipment_order_line -> release_shipment_order"
    - "Shipments can move between: site<->site, remote_warehouse<->site, remote_warehouse<->remote_warehouse"
    - "shipment_order_id is always an INTEGER (e.g., 34), never a string like 'SO-34'"

inventory:
  keywords: [inventory, stock, qty, quantity, onhand, available, reserve]
  facts:
    - "Use get_inventory_stock to check ALL warehouses at a site"
    - "Use search_inventory_by_warehouse for ONE specific warehouse only"
    - "Parameter is 'part_no' (not 'part' or 'part_number')"

orders:
  keywords: [order, customer order, sales order, line, order line]
  facts:
    - "Order numbers with asterisks like '*1063' should include the asterisk"
    - "Parameter is 'order_no' (not 'order_number' or 'orderNo')"
```

---

### Step 3: Add Semantic Memory to MemoryManager

**File:** `src/tools/memory_manager.py`

#### 3a. Add import at top:

```python
import yaml
```

#### 3b. Add to `__init__` method (around line 46):

```python
self.semantic_memory: Dict[str, Dict] = {}
```

#### 3c. Add new methods after `load_memories()` (around line 77):

```python
def load_semantic_memory(self, config_path: str = "./config/ifs_semantic.yaml") -> None:
    """Load semantic memory from YAML config."""
    if not os.path.exists(config_path):
        self.semantic_memory = {}
        return
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            self.semantic_memory = yaml.safe_load(f) or {}
    except Exception:
        self.semantic_memory = {}

def retrieve_semantic_facts(self, query: str, max_facts: int = 5) -> List[str]:
    """Retrieve semantic facts relevant to the query based on keyword matching."""
    if not self.semantic_memory:
        return []

    query_words = set(query.lower().split())
    matched_facts: List[str] = []

    for category, data in self.semantic_memory.items():
        if not isinstance(data, dict):
            continue
        keywords = set(word.lower() for word in data.get('keywords', []))
        if query_words & keywords:  # Any keyword overlap
            matched_facts.extend(data.get('facts', []))

    return matched_facts[:max_facts]
```

#### 3d. Update `format_memories_for_prompt()` to include semantic facts (around line 350):

```python
def format_memories_for_prompt(
    self,
    query: str,
    available_tool_names: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
) -> str:
    parts = []

    # 1. Semantic facts (domain knowledge) - NEW
    semantic_facts = self.retrieve_semantic_facts(query)
    if semantic_facts:
        parts.append("## Domain Knowledge")
        for fact in semantic_facts:
            parts.append(f"- {fact}")
        parts.append("")

    # 2. Episodic memories (existing code continues below)
    episodic_memories = self.retrieve_relevant_episodic_memories(
        query=query,
        dataset_name=dataset_name,
    )
    # ... rest of existing implementation ...
```

#### 3e. Delete `load_seed_memories()` method (lines 482-561)

This method is no longer needed since procedural rules are now embedded in tool definitions.

---

### Step 4: Update Flask App

**File:** `src/app_flask.py`

#### 4a. Update imports (around line 20):

```python
from tools.mcp_tool_registry import build_tool_prompt, TOOL_REGISTRY, get_tools_for_intent
```

#### 4b. Replace seed memory loading with semantic memory loading in `initialize()` (lines 72-76):

**REPLACE this:**
```python
if _tool_manager.memory_manager is not None:
    rules_loaded = _tool_manager.memory_manager.load_seed_memories()
    if rules_loaded > 0:
        print(f"  📚 Loaded {rules_loaded} procedural rules from seed memories")
```

**WITH this:**
```python
if _tool_manager.memory_manager is not None:
    _tool_manager.memory_manager.load_semantic_memory()
    print("  📚 Loaded semantic memory from config")
```

#### 4c. (Optional) Add smart tool filtering in `process_message_streaming()` after memory retrieval (around line 270):

```python
# Smart tool filtering - only include relevant tools
relevant_tools = get_tools_for_intent(user_message)
if relevant_tools:
    relevant_tool_names = [t.name for t in relevant_tools]
    # Use filtered tools for memory retrieval
    tool_names = relevant_tool_names
else:
    tool_names = list(TOOL_REGISTRY.keys())
```

---

### Step 5: Clean Up

**Delete file:** `cache/memory/seed_memories.json`

This file is no longer needed since:
- Procedural rules are now embedded in tool definitions (Step 1)
- Semantic facts are in `config/ifs_semantic.yaml` (Step 2)

---

## Files Summary

| File | Action |
|------|--------|
| `src/tools/mcp_tool_registry.py` | Add `rules` field to ToolSummary, update 3 shipment tools, modify `get_tool_summaries_for_prompt()` |
| `src/tools/memory_manager.py` | Add `load_semantic_memory()`, `retrieve_semantic_facts()`, update `format_memories_for_prompt()`, delete `load_seed_memories()` |
| `src/app_flask.py` | Replace seed loading with semantic loading, optionally add smart tool filtering |
| `config/ifs_semantic.yaml` | **CREATE** new file |
| `cache/memory/seed_memories.json` | **DELETE** |

---

## Testing

After implementation, test these scenarios:

1. **"Create a shipment order to move 100 of part 10106105 from A110 to A205"** - should show procedural rules in tool output

2. **"What is the inventory for part 10106105 at site AC?"** - should retrieve semantic facts about inventory

3. **"Show me past due orders"** - episodic memory should still work
