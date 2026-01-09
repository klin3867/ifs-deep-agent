#!/usr/bin/env python3
"""
Test semantic retrieval for MCPToolRetriever and KnowledgeRetriever.
Compares actual results to expected results for hypothetical user prompts.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tools.mcp_tool_registry import MCPToolRetriever, TOOL_REGISTRY
from tools.memory_manager import MemoryManager


def test_inventory_query():
    """Test Scenario 1: Inventory Query"""
    print("=" * 60)
    print("SCENARIO 1: Inventory Query")
    print("=" * 60)
    print("User prompt: 'What is the inventory for part 10106105 at site AC?'")
    print()

    # Simulate tool search query (what LLM would emit)
    tool_search_query = "inventory stock part"

    # Test MCPToolRetriever
    print("--- MCPToolRetriever Results ---")
    retriever = MCPToolRetriever.get_instance()
    tools_with_scores = retriever.retrieve_with_scores(tool_search_query, top_k=10)

    print(f"Query: '{tool_search_query}'")
    print(f"Top tools found:")
    for name, score in tools_with_scores:
        print(f"  {score:.3f} - {name}")

    # Check expected tools are in results
    expected_tools = ["get_inventory_stock", "search_inventory_by_warehouse"]
    found = [name for name, _ in tools_with_scores[:5]]
    for expected in expected_tools:
        status = "✅" if expected in found else "❌"
        print(f"{status} Expected '{expected}' in top 5: {expected in found}")

    print()

    # Test KnowledgeRetriever
    print("--- KnowledgeRetriever Results ---")
    mm = MemoryManager()
    mm.load_knowledge_base("./config/ifs_knowledge.yaml", use_semantic=True)

    knowledge = mm.retrieve_relevant_knowledge(
        query=tool_search_query,
        tool_names=[name for name, _ in tools_with_scores[:5]]
    )

    print(f"Procedural rules retrieved ({len(knowledge['procedural_rules'])}):")
    for rule in knowledge['procedural_rules']:
        print(f"  - {rule[:80]}...")

    print(f"\nSemantic facts retrieved ({len(knowledge['semantic_facts'])}):")
    for fact in knowledge['semantic_facts']:
        print(f"  - {fact[:80]}...")

    print(f"\nError corrections ({len(knowledge['error_corrections'])}):")
    for err in knowledge['error_corrections']:
        print(f"  - {err[:80]}...")

    # Verify key rules are present
    expected_rules = [
        "part_no (not part or part_number)",
        "Site parameter is 'site'",
    ]
    all_rules_text = " ".join(knowledge['procedural_rules'])
    print("\n--- Expected Rules Check ---")
    for expected in expected_rules:
        status = "✅" if expected in all_rules_text else "❌"
        print(f"{status} Expected rule containing '{expected[:40]}...'")

    return True


def test_shipment_creation():
    """Test Scenario 2: Shipment Order Creation"""
    print("\n" + "=" * 60)
    print("SCENARIO 2: Shipment Order Creation")
    print("=" * 60)
    print("User prompt: 'Create a shipment order to move 100000 from A105 to A205'")
    print()

    # Simulate tool search query
    tool_search_query = "create shipment order move transfer"

    # Test MCPToolRetriever
    print("--- MCPToolRetriever Results ---")
    retriever = MCPToolRetriever.get_instance()
    tools_with_scores = retriever.retrieve_with_scores(tool_search_query, top_k=10)

    print(f"Query: '{tool_search_query}'")
    print(f"Top tools found:")
    for name, score in tools_with_scores:
        print(f"  {score:.3f} - {name}")

    # Check expected tools
    expected_tools = ["create_shipment_order", "add_shipment_order_line", "release_shipment_order"]
    found = [name for name, _ in tools_with_scores[:5]]
    print("\n--- Expected Tools Check ---")
    for expected in expected_tools:
        status = "✅" if expected in found else "❌"
        print(f"{status} Expected '{expected}' in top 5: {expected in found}")

    print()

    # Test KnowledgeRetriever
    print("--- KnowledgeRetriever Results ---")
    mm = MemoryManager()
    mm.load_knowledge_base("./config/ifs_knowledge.yaml", use_semantic=True)

    knowledge = mm.retrieve_relevant_knowledge(
        query=tool_search_query,
        tool_names=[name for name, _ in tools_with_scores[:5]]
    )

    print(f"Procedural rules retrieved ({len(knowledge['procedural_rules'])}):")
    for rule in knowledge['procedural_rules']:
        print(f"  - {rule[:100]}...")

    print(f"\nSemantic facts retrieved ({len(knowledge['semantic_facts'])}):")
    for fact in knowledge['semantic_facts']:
        print(f"  - {fact[:100]}...")

    print(f"\nError corrections ({len(knowledge['error_corrections'])}):")
    for err in knowledge['error_corrections']:
        print(f"  - {err[:100]}...")

    # Verify critical rules for this scenario
    expected_patterns = [
        "from_warehouse",        # Correct MCP parameter name
        "to_warehouse",          # Correct MCP parameter name
        "shipment_order_id",     # Integer rule
        "3 steps",               # Workflow
    ]
    all_text = " ".join(knowledge['procedural_rules'] + knowledge['semantic_facts'])

    print("\n--- Critical Knowledge Check ---")
    for pattern in expected_patterns:
        status = "✅" if pattern in all_text else "❌"
        print(f"{status} Contains '{pattern}': {pattern in all_text}")

    return True


if __name__ == "__main__":
    print("\n🧪 SEMANTIC RETRIEVAL VERIFICATION TEST\n")

    test_inventory_query()
    test_shipment_creation()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
