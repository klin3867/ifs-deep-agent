"""
Compact tool registry for LLM context with lazy-load schema pattern.

Instead of sending full JSON schemas (~500 tokens each), we send
one-line summaries (~20 tokens each). The LLM uses meta-tools to:
1. Browse available tools (see summaries below)
2. Call get_mcp_tool_schema(tool_name) to get full parameter schema
3. Call call_mcp_tool(tool_name, arguments) with correct parameters

This follows the RestBench lazy-load pattern for token efficiency.

Token Savings:
- Full schemas: ~500 tokens x 60 tools = ~30,000 tokens per call
- Summaries + meta-tools: ~1,400 tokens per call
- Per-tool schema on demand: ~200-500 tokens
- Savings: 90-95% reduction in token usage
"""

from typing import Any, Dict, List, NamedTuple, Optional, Set


# =============================================================================
# META-TOOLS - Used by LLM to discover schemas on-demand
# =============================================================================

MCP_META_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_mcp_tool_schema",
            "description": "Get the full parameter schema for an MCP tool before calling it. Use this to learn the exact parameter names and types required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Name of the tool to get schema for (e.g., 'add_order_line', 'search_orders')"
                    }
                },
                "required": ["tool_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_mcp_tool",
            "description": "Execute an MCP tool with arguments. Always use get_mcp_tool_schema first to learn the correct parameter names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Name of the tool to execute"
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Tool arguments as key-value pairs matching the schema",
                        "additionalProperties": True
                    }
                },
                "required": ["tool_name", "arguments"]
            }
        }
    }
]


def get_meta_tools() -> List[Dict[str, Any]]:
    """Return the meta-tools for MCP lazy-load pattern."""
    return MCP_META_TOOLS


def get_meta_tool_names() -> List[str]:
    """Return names of meta-tools."""
    return [t["function"]["name"] for t in MCP_META_TOOLS]


class ToolSummary(NamedTuple):
    """Compact tool metadata for LLM decision-making."""
    name: str
    server: str
    summary: str  # One line, <100 chars
    category: str  # orders, customers, inventory, shipments, planning, reservations
    mutates: bool  # True for create/update/delete
    use_when: str  # When would someone need this?


# =============================================================================
# TOOL REGISTRY - 59 tools organized by category
# =============================================================================

TOOL_REGISTRY: Dict[str, ToolSummary] = {
    
    # =========================================================================
    # SYSTEM (1 tool)
    # =========================================================================
    "health_check": ToolSummary(
        name="health_check",
        server="customer",
        summary="Verify IFS connection and server configuration",
        category="system",
        mutates=False,
        use_when="User wants to verify connection, troubleshoot issues, or check server status",
    ),
    
    # =========================================================================
    # CUSTOMERS (1 tool)
    # =========================================================================
    "search_customers": ToolSummary(
        name="search_customers",
        server="customer",
        summary="Find customers by name or customer number",
        category="customers",
        mutates=False,
        use_when="User wants to look up a customer, find customer details, or search by name",
    ),
    
    # =========================================================================
    # PARTS (2 tools)
    # =========================================================================
    "get_sales_part": ToolSummary(
        name="get_sales_part",
        server="customer",
        summary="Get full part details including UoM, price, and description",
        category="parts",
        mutates=False,
        use_when="User needs part details, unit of measure, pricing, or catalog info",
    ),
    "search_sales_parts": ToolSummary(
        name="search_sales_parts",
        server="customer",
        summary="Search for parts by catalog number or description",
        category="parts",
        mutates=False,
        use_when="User wants to find a part, search catalog, or look up parts",
    ),
    
    # =========================================================================
    # ORDERS - Read (14 tools)
    # =========================================================================
    "search_orders": ToolSummary(
        name="search_orders",
        server="customer",
        summary="Find order headers by customer, order number, or status",
        category="orders",
        mutates=False,
        use_when="User wants to find orders, search by customer or status",
    ),
    "search_order_lines": ToolSummary(
        name="search_order_lines",
        server="customer",
        summary="Search order lines across all orders by customer, part, or status",
        category="orders",
        mutates=False,
        use_when="User wants to find specific order lines, search across orders",
    ),
    "get_order_details": ToolSummary(
        name="get_order_details",
        server="customer",
        summary="Get full order with header and all line items",
        category="orders",
        mutates=False,
        use_when="User asks for details of a specific order",
    ),
    "get_recent_order_lines": ToolSummary(
        name="get_recent_order_lines",
        server="customer",
        summary="Get all order lines for orders entered in the last N days",
        category="orders",
        mutates=False,
        use_when="User wants recent orders, batch view of order lines by date",
    ),
    "list_past_due_lines": ToolSummary(
        name="list_past_due_lines",
        server="customer",
        summary="Get order lines past their promised delivery date that are still open",
        category="orders",
        mutates=False,
        use_when="User asks about overdue orders, late deliveries, past due items",
    ),
    "search_customer_orders": ToolSummary(
        name="search_customer_orders",
        server="planning",
        summary="Search customer order headers by customer, status, or date range",
        category="orders",
        mutates=False,
        use_when="User wants to find customer orders, filter by status or customer",
    ),
    "search_customer_order_lines": ToolSummary(
        name="search_customer_order_lines",
        server="planning",
        summary="Search order lines for demand visibility across orders",
        category="orders",
        mutates=False,
        use_when="User needs to see demand/order lines for planning purposes",
    ),
    "get_customer_order_details": ToolSummary(
        name="get_customer_order_details",
        server="planning",
        summary="Get customer order header plus all line items and statuses",
        category="orders",
        mutates=False,
        use_when="User needs complete order view for planning analysis",
    ),
    "list_past_due_customer_order_lines": ToolSummary(
        name="list_past_due_customer_order_lines",
        server="planning",
        summary="List order lines past due with dollar totals and days late",
        category="orders",
        mutates=False,
        use_when="User asks about late orders, overdue shipments, past due with dollar impact",
    ),
    "get_recent_customer_order_lines": ToolSummary(
        name="get_recent_customer_order_lines",
        server="planning",
        summary="Get order lines for orders entered in last N days (batched)",
        category="orders",
        mutates=False,
        use_when="User needs recent order activity, lines by order date entered",
    ),
    "list_reservable_customer_order_lines": ToolSummary(
        name="list_reservable_customer_order_lines",
        server="planning",
        summary="List order lines available for reservation at a site",
        category="orders",
        mutates=False,
        use_when="User wants to see what can be reserved, released lines needing stock",
    ),
    
    # =========================================================================
    # ORDERS - Write (5 tools)
    # =========================================================================
    "create_order": ToolSummary(
        name="create_order",
        server="customer",
        summary="Create a new customer order",
        category="orders",
        mutates=True,
        use_when="User wants to create, place, or start a new order",
    ),
    "add_order_line": ToolSummary(
        name="add_order_line",
        server="customer",
        summary="Add a line item to an existing order",
        category="orders",
        mutates=True,
        use_when="User wants to add a product/part to an order",
    ),
    "update_order_line": ToolSummary(
        name="update_order_line",
        server="customer",
        summary="Update order line quantity or delivery dates",
        category="orders",
        mutates=True,
        use_when="User wants to change qty, update delivery date, modify order line",
    ),
    "cancel_order": ToolSummary(
        name="cancel_order",
        server="customer",
        summary="Cancel an entire customer order",
        category="orders",
        mutates=True,
        use_when="User wants to cancel a full order",
    ),
    "cancel_order_line": ToolSummary(
        name="cancel_order_line",
        server="customer",
        summary="Cancel a specific order line",
        category="orders",
        mutates=True,
        use_when="User wants to cancel just one line from an order",
    ),
    
    # =========================================================================
    # INVENTORY (6 tools)
    # =========================================================================
    "get_inventory_stock": ToolSummary(
        name="get_inventory_stock",
        server="planning",
        summary="Get inventory stock levels by part, site, and warehouse",
        category="inventory",
        mutates=False,
        use_when="User wants to check stock, see inventory levels, query on-hand qty",
    ),
    "search_inventory_by_warehouse": ToolSummary(
        name="search_inventory_by_warehouse",
        server="planning",
        summary="Search inventory across warehouses with availability details",
        category="inventory",
        mutates=False,
        use_when="User wants to find stock in specific warehouse, compare warehouses",
    ),
    "get_inventory_by_handling_unit": ToolSummary(
        name="get_inventory_by_handling_unit",
        server="planning",
        summary="Get inventory details for a specific handling unit (pallet/case)",
        category="inventory",
        mutates=False,
        use_when="User asks about specific HU, pallet, or handling unit contents",
    ),
    "check_stock_availability": ToolSummary(
        name="check_stock_availability",
        server="planning",
        summary="Check if sufficient stock exists for a part/qty at a site",
        category="inventory",
        mutates=False,
        use_when="User asks do we have enough, can we fulfill, availability check",
    ),
    "get_reservable_stock": ToolSummary(
        name="get_reservable_stock",
        server="planning",
        summary="Get stock available for reservation with warehouse/tier details",
        category="inventory",
        mutates=False,
        use_when="User wants to see what inventory can be reserved, stock by tier",
    ),
    "get_available_handling_units": ToolSummary(
        name="get_available_handling_units",
        server="planning",
        summary="List available handling units (pallets) for a part at a site",
        category="inventory",
        mutates=False,
        use_when="User wants to see pallets, HUs available, handling unit list",
    ),
    
    # =========================================================================
    # RESERVATIONS (13 tools)
    # =========================================================================
    "list_reservable_lines": ToolSummary(
        name="list_reservable_lines",
        server="customer",
        summary="List order lines available for inventory reservation",
        category="reservations",
        mutates=False,
        use_when="User wants to see what order lines can be reserved",
    ),
    "get_reservation_status": ToolSummary(
        name="get_reservation_status",
        server="customer",
        summary="Get qty reserved/available/picked for a specific order line",
        category="reservations",
        mutates=False,
        use_when="User asks about reservation status, how much is reserved for a line",
    ),
    "reserve_order_lines_for_site": ToolSummary(
        name="reserve_order_lines_for_site",
        server="customer",
        summary="Reserve inventory for order lines from a specific site",
        category="reservations",
        mutates=True,
        use_when="User wants to reserve stock, allocate inventory to order lines",
    ),
    "plan_reservation": ToolSummary(
        name="plan_reservation",
        server="planning",
        summary="Create reservation plan for a part/qty with warehouse cascade",
        category="reservations",
        mutates=False,
        use_when="User wants to plan a reservation, see where stock would come from",
    ),
    "plan_customer_order_reservation": ToolSummary(
        name="plan_customer_order_reservation",
        server="planning",
        summary="Plan reservation for a customer order line with FEFO logic",
        category="reservations",
        mutates=False,
        use_when="User wants reservation plan for specific order line",
    ),
    "execute_reservation": ToolSummary(
        name="execute_reservation",
        server="planning",
        summary="Execute a previously planned reservation (two-phase commit)",
        category="reservations",
        mutates=True,
        use_when="User approves and wants to execute a reservation plan",
    ),
    "check_shipment_order_availability": ToolSummary(
        name="check_shipment_order_availability",
        server="planning",
        summary="Check stock availability for all lines on a shipment order",
        category="reservations",
        mutates=False,
        use_when="User asks if shipment can be fulfilled, availability for shipment",
    ),
    "get_shipment_line_available_stock": ToolSummary(
        name="get_shipment_line_available_stock",
        server="planning",
        summary="Get available stock for a specific shipment line",
        category="reservations",
        mutates=False,
        use_when="User wants to see what stock is available for a shipment line",
    ),
    "reserve_shipment_line_handling_unit": ToolSummary(
        name="reserve_shipment_line_handling_unit",
        server="planning",
        summary="Reserve a full handling unit for a shipment line",
        category="reservations",
        mutates=True,
        use_when="User wants to reserve a pallet/HU for shipment",
    ),
    "reserve_shipment_line_partial": ToolSummary(
        name="reserve_shipment_line_partial",
        server="planning",
        summary="Reserve partial quantity from a location for shipment",
        category="reservations",
        mutates=True,
        use_when="User wants to reserve less than full HU, partial reservation",
    ),
    "plan_shipment_reservation_fefo": ToolSummary(
        name="plan_shipment_reservation_fefo",
        server="planning",
        summary="Plan shipment reservation using FEFO (First Expire First Out) logic",
        category="reservations",
        mutates=False,
        use_when="User wants FEFO-based reservation plan for shipment",
    ),
    "execute_shipment_reservation_plan": ToolSummary(
        name="execute_shipment_reservation_plan",
        server="planning",
        summary="Execute a shipment reservation plan (two-phase commit)",
        category="reservations",
        mutates=True,
        use_when="User approves and wants to execute shipment reservation",
    ),
    "reserve_shipment_order": ToolSummary(
        name="reserve_shipment_order",
        server="planning",
        summary="Reserve inventory for an entire shipment order",
        category="reservations",
        mutates=True,
        use_when="User wants to reserve all lines on a shipment",
    ),
    
    # =========================================================================
    # SHIPMENTS (5 tools)
    # =========================================================================
    "create_shipment_order": ToolSummary(
        name="create_shipment_order",
        server="planning",
        summary="Create a new shipment order",
        category="shipments",
        mutates=True,
        use_when="User wants to create a shipment, start new shipment order",
    ),
    "add_shipment_order_line": ToolSummary(
        name="add_shipment_order_line",
        server="planning",
        summary="Add a line item to a shipment order",
        category="shipments",
        mutates=True,
        use_when="User wants to add parts to a shipment",
    ),
    "release_shipment_order": ToolSummary(
        name="release_shipment_order",
        server="planning",
        summary="Release a shipment order for processing",
        category="shipments",
        mutates=True,
        use_when="User wants to release/approve shipment for warehouse",
    ),
    "list_shipment_orders": ToolSummary(
        name="list_shipment_orders",
        server="planning",
        summary="List shipment orders by status, customer, or date",
        category="shipments",
        mutates=False,
        use_when="User wants to see shipments, list shipment orders",
    ),
    "get_shipment_order_details": ToolSummary(
        name="get_shipment_order_details",
        server="planning",
        summary="Get shipment order header and all line details",
        category="shipments",
        mutates=False,
        use_when="User asks for details of a specific shipment",
    ),
    
    # =========================================================================
    # PLANNING (8 tools)
    # =========================================================================
    "generate_planning_snapshot": ToolSummary(
        name="generate_planning_snapshot",
        server="planning",
        summary="Generate MRP planning snapshot for a part (returns SnapshotId)",
        category="planning",
        mutates=False,
        use_when="User wants MRP analysis, needs to run planning for a part",
    ),
    "get_part_supply_demand": ToolSummary(
        name="get_part_supply_demand",
        server="planning",
        summary="Get supply/demand breakdown from planning snapshot",
        category="planning",
        mutates=False,
        use_when="User wants to see supply vs demand, what is coming in/out",
    ),
    "get_plannable_summary": ToolSummary(
        name="get_plannable_summary",
        server="planning",
        summary="Get plannable quantities by date from planning snapshot",
        category="planning",
        mutates=False,
        use_when="User wants planning summary, qty available by date",
    ),
    "list_shortages": ToolSummary(
        name="list_shortages",
        server="planning",
        summary="List parts with shortages (demand exceeds supply)",
        category="planning",
        mutates=False,
        use_when="User asks about shortages, what is short, supply gaps",
    ),
    "get_pegging_details": ToolSummary(
        name="get_pegging_details",
        server="planning",
        summary="Get pegging - which supplies are committed to which demands",
        category="planning",
        mutates=False,
        use_when="User asks what order is using which supply, pegging info",
    ),
    "get_reserved_vs_unreserved": ToolSummary(
        name="get_reserved_vs_unreserved",
        server="planning",
        summary="Compare reserved vs unreserved quantities for planning",
        category="planning",
        mutates=False,
        use_when="User wants to see what is reserved vs available",
    ),
    "get_order_references": ToolSummary(
        name="get_order_references",
        server="planning",
        summary="Get order references from supply/demand - what drives demand",
        category="planning",
        mutates=False,
        use_when="User asks what orders are creating demand/supply",
    ),
    "analyze_unreserved_demand_by_warehouse": ToolSummary(
        name="analyze_unreserved_demand_by_warehouse",
        server="planning",
        summary="Strategic analysis: unreserved demand vs warehouse availability",
        category="planning",
        mutates=False,
        use_when="User asks about what needs reservation, sourcing analysis",
    ),
    
    # =========================================================================
    # EXCEPTIONS (7 tools)
    # =========================================================================
    "list_inventory_exceptions": ToolSummary(
        name="list_inventory_exceptions",
        server="planning",
        summary="List MRP inventory exceptions by type",
        category="exceptions",
        mutates=False,
        use_when="User asks about inventory exceptions, MRP alerts, planning issues",
    ),
    "get_parts_needing_orders": ToolSummary(
        name="get_parts_needing_orders",
        server="planning",
        summary="Get parts where supply + on-hand will not cover demand",
        category="exceptions",
        mutates=False,
        use_when="User asks what parts need ordering, supply shortfall",
    ),
    "get_negative_projected_onhand": ToolSummary(
        name="get_negative_projected_onhand",
        server="planning",
        summary="Get parts that will go negative in future",
        category="exceptions",
        mutates=False,
        use_when="User asks about future stockouts, projected negative inventory",
    ),
    "get_excess_orders": ToolSummary(
        name="get_excess_orders",
        server="planning",
        summary="Get parts with more supply than needed (excess)",
        category="exceptions",
        mutates=False,
        use_when="User asks about excess inventory, over-ordering",
    ),
    "get_past_due_exceptions": ToolSummary(
        name="get_past_due_exceptions",
        server="planning",
        summary="Get parts with past due supply or demand orders",
        category="exceptions",
        mutates=False,
        use_when="User asks about past due exceptions, late supply/demand",
    ),
    "get_parts_with_no_demand": ToolSummary(
        name="get_parts_with_no_demand",
        server="planning",
        summary="Get parts in inventory with no demand (potential obsolete)",
        category="exceptions",
        mutates=False,
        use_when="User asks about dead stock, no demand parts, obsolete inventory",
    ),
    "confirm_exception": ToolSummary(
        name="confirm_exception",
        server="planning",
        summary="Mark an MRP exception as reviewed/confirmed",
        category="exceptions",
        mutates=True,
        use_when="User wants to acknowledge/confirm an exception was reviewed",
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_tool_summaries_for_prompt(
    categories: Optional[List[str]] = None,
    exclude_categories: Optional[List[str]] = None,
    mutating_only: bool = False,
    read_only: bool = False,
) -> str:
    """
    Generate compact tool list for LLM prompt.
    
    Args:
        categories: Only include tools from these categories
        exclude_categories: Exclude tools from these categories
        mutating_only: Only include tools that mutate state
        read_only: Only include tools that do not mutate state
        
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
    
    return "\n".join(lines)


def get_tools_by_category(category: str) -> List[ToolSummary]:
    """Get all tools in a category."""
    return [t for t in TOOL_REGISTRY.values() if t.category == category]


def get_tool_for_execution(name: str) -> Optional[ToolSummary]:
    """Look up tool by name for execution."""
    return TOOL_REGISTRY.get(name)


def get_all_categories() -> Set[str]:
    """Get all unique categories."""
    return set(t.category for t in TOOL_REGISTRY.values())


def get_category_counts() -> Dict[str, int]:
    """Get count of tools per category."""
    counts: Dict[str, int] = {}
    for t in TOOL_REGISTRY.values():
        counts[t.category] = counts.get(t.category, 0) + 1
    return counts


def get_mutating_tool_names() -> List[str]:
    """Get names of all tools that mutate state."""
    return [t.name for t in TOOL_REGISTRY.values() if t.mutates]


def search_tools(query: str) -> List[ToolSummary]:
    """
    Search tools by query string (checks name, summary, use_when).
    
    Useful for finding relevant tools based on user intent.
    """
    query_lower = query.lower()
    results = []
    for tool in TOOL_REGISTRY.values():
        if (query_lower in tool.name.lower() or 
            query_lower in tool.summary.lower() or
            query_lower in tool.use_when.lower()):
            results.append(tool)
    return results


def get_tools_for_intent(intent: str) -> List[ToolSummary]:
    """
    Get tools likely relevant for a given user intent.
    
    Uses keyword matching against use_when field.
    """
    intent_lower = intent.lower()
    scored_tools = []
    for tool in TOOL_REGISTRY.values():
        use_when_words = set(tool.use_when.lower().split())
        intent_words = set(intent_lower.split())
        overlap = len(use_when_words & intent_words)
        if overlap > 0:
            scored_tools.append((overlap, tool))
    scored_tools.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored_tools[:10]]


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

CATEGORY_DESCRIPTION = """
Available tool categories:
- system: Server health and connection checks
- customers: Customer lookup and search
- parts: Part/catalog information
- orders: Customer order management (search, create, update, cancel)
- inventory: Stock levels, warehouse queries
- reservations: Reserve inventory for orders/shipments
- shipments: Shipment order management
- planning: MRP planning, supply/demand analysis
- exceptions: MRP exceptions and alerts
"""


def build_tool_prompt(
    categories: Optional[List[str]] = None,
    include_categories_description: bool = True,
    lazy_load: bool = False,
) -> str:
    """
    Build a compact prompt section listing available tools.
    
    Args:
        categories: Filter to specific categories
        include_categories_description: Include category descriptions
        lazy_load: If True, include meta-tool instructions for lazy schema loading
    
    This replaces the full JSON schema approach with compact summaries.
    """
    parts = []
    
    if include_categories_description:
        parts.append(CATEGORY_DESCRIPTION.strip())
        parts.append("")
    parts.append("Available tools:")
    parts.append(get_tool_summaries_for_prompt(categories=categories))
    return "\n".join(parts)


# =============================================================================
# STATISTICS
# =============================================================================

def print_registry_stats():
    """Print statistics about the tool registry."""
    total = len(TOOL_REGISTRY)
    by_category = get_category_counts()
    mutating = len(get_mutating_tool_names())
    
    print("\n=== Tool Registry Statistics ===")
    print(f"Total tools: {total}")
    print(f"Mutating tools: {mutating}")
    print(f"Read-only tools: {total - mutating}")
    print("\nBy category:")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat}: {count}")
    
    full_schema_tokens = total * 500
    summary_tokens = total * 20
    print("\nToken savings:")
    print(f"  Full schemas: ~{full_schema_tokens:,} tokens")
    print(f"  Summaries: ~{summary_tokens:,} tokens")
    print(f"  Savings: ~{int((1 - summary_tokens/full_schema_tokens)*100)}%")


# =============================================================================
# REDUNDANCY NOTES
# =============================================================================
"""
Potentially Redundant Tools (for future consolidation):

1. Order search tools - Overlap between customer and planning servers:
   - search_orders (customer) vs search_customer_orders (planning)
   - search_order_lines (customer) vs search_customer_order_lines (planning)
   - get_order_details (customer) vs get_customer_order_details (planning)
   - get_recent_order_lines (customer) vs get_recent_customer_order_lines (planning)
   - list_past_due_lines (customer) vs list_past_due_customer_order_lines (planning)
   
   Recommendation: The planning server versions have more features (dollar totals,
   days late calculation). Consider deprecating customer server versions or routing
   based on whether user needs basic search vs planning analysis.

2. Reservable lines:
   - list_reservable_lines (customer) vs list_reservable_customer_order_lines (planning)
   
   Recommendation: Consolidate to planning server version which has more context.

3. Planning snapshot tools require a SnapshotId from generate_planning_snapshot.
   Consider: Auto-generating snapshot if user asks for supply/demand analysis
   without having generated one first.
"""


# =============================================================================
# SEMANTIC TOOL RETRIEVER
# =============================================================================

class MCPToolRetriever:
    """
    Semantic retriever for MCP tools using SentenceTransformer.
    Similar to native ToolRetriever but for the 59 IFS Cloud tools.
    """

    _instance = None  # Singleton to avoid reloading embeddings

    def __init__(self, model_path: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer, util
        import torch

        self.model_path = model_path
        self.embedder = SentenceTransformer(model_path)
        self._util = util  # Store for later use

        # Build corpus from TOOL_REGISTRY
        self.corpus = []
        self.corpus_to_tool = {}

        for name, tool in TOOL_REGISTRY.items():
            # Index content: name + summary + use_when + category
            index_text = f"{name}, {tool.summary}, {tool.use_when}, category: {tool.category}"
            self.corpus.append(index_text)
            self.corpus_to_tool[index_text] = name

        # Pre-compute embeddings (one-time cost)
        self.corpus_embeddings = self.embedder.encode(
            self.corpus,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        print(f"MCPToolRetriever: Indexed {len(self.corpus)} tools")

    @classmethod
    def get_instance(cls, model_path: str = "BAAI/bge-small-en-v1.5") -> "MCPToolRetriever":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """
        Retrieve top_k tool names most relevant to query.

        Args:
            query: Natural language search query
            top_k: Number of tools to return

        Returns:
            List of tool names (not full schemas)
        """
        query_embedding = self.embedder.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        hits = self._util.semantic_search(
            query_embedding,
            self.corpus_embeddings,
            top_k=top_k,
            score_function=self._util.cos_sim
        )

        tool_names = []
        for hit in hits[0]:
            corpus_text = self.corpus[hit['corpus_id']]
            tool_name = self.corpus_to_tool[corpus_text]
            tool_names.append(tool_name)

        return tool_names

    def retrieve_with_scores(self, query: str, top_k: int = 10) -> List[tuple]:
        """
        Retrieve top_k tools with similarity scores.

        Returns:
            List of (tool_name, score) tuples
        """
        query_embedding = self.embedder.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        hits = self._util.semantic_search(
            query_embedding,
            self.corpus_embeddings,
            top_k=top_k,
            score_function=self._util.cos_sim
        )

        results = []
        for hit in hits[0]:
            corpus_text = self.corpus[hit['corpus_id']]
            tool_name = self.corpus_to_tool[corpus_text]
            results.append((tool_name, hit['score']))

        return results


def get_full_tool_schemas(tool_names: List[str], mcp_caller) -> List[Dict]:
    """
    Get full JSON schemas for specified tools from MCP cache.

    Args:
        tool_names: List of tool names to get schemas for
        mcp_caller: MCPToolCaller instance with cached tool schemas

    Returns:
        List of full tool schemas (OpenAI function format)
    """
    schemas = []
    for name in tool_names:
        schema = mcp_caller.get_tool_schema(name)
        if "error" not in schema:
            schemas.append(schema)
    return schemas


if __name__ == "__main__":
    print_registry_stats()
    print("\n" + "="*60)
    print("\nExample prompt (orders category only):")
    print(build_tool_prompt(categories=["orders"]))
