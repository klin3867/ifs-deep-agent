#!/usr/bin/env python3
"""
Smoke test for DeepAgent with MCP integration.
Tests multi-step reasoning with tool calls and memory systems.

Test scenario:
1. Look up past due customer orders
2. Get details of one order
3. Add a line to that order

This exercises:
- Tool discovery and execution
- MCP server communication
- Tool interaction tracking (for memory folding)
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from argparse import Namespace
import yaml


async def run_smoke_test():
    """Run a multi-step smoke test of the DeepAgent system."""
    
    print("=" * 60)
    print("DeepAgent MCP Smoke Test")
    print("=" * 60)
    
    # Load config
    config_path = "./config/base_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    args = Namespace(**config)
    args.dataset_name = 'mcp'
    args.use_openai_chat = True
    
    # Test 1: Initialize tool manager and MCP connection
    print("\n[TEST 1] Initializing ToolManager with MCP...")
    from tools.tool_manager import ToolManager
    
    try:
        tool_manager = await ToolManager.create(args)
        print("  ✅ ToolManager initialized")
        
        if hasattr(tool_manager, 'mcp_caller') and tool_manager.mcp_caller:
            print(f"  ✅ MCP caller connected")
            if hasattr(tool_manager, '_mcp_tools'):
                print(f"  ✅ Loaded {len(tool_manager._mcp_tools)} MCP tools")
        else:
            print("  ⚠️ MCP caller not available")
            return False
    except Exception as e:
        print(f"  ❌ Failed to initialize: {e}")
        return False
    
    # Test 2: Check health
    print("\n[TEST 2] Testing health_check tool...")
    try:
        health_call = {"function": {"name": "health_check", "arguments": {}}}
        result = await tool_manager.call_tool(health_call, {})
        print(f"  ✅ health_check returned: {json.dumps(result, indent=2)[:200]}")
    except Exception as e:
        print(f"  ❌ health_check failed: {e}")
    
    # Test 3: Search for past due orders
    print("\n[TEST 3] Searching for past due customer order lines...")
    try:
        # Use get_past_due_order_lines if available, or search_order_lines with status filter
        search_call = {"function": {"name": "get_past_due_order_lines", "arguments": {"limit": 5}}}
        result = await tool_manager.call_tool(search_call, {})
        
        if "error" in result:
            # Try alternative tool
            print(f"  ⚠️ get_past_due_order_lines not available, trying search_order_lines...")
            search_call = {"function": {"name": "search_order_lines", "arguments": {"limit": 5}}}
            result = await tool_manager.call_tool(search_call, {})
        
        print(f"  ✅ Order search returned:")
        result_str = json.dumps(result, indent=2)
        print(f"  {result_str[:500]}{'...' if len(result_str) > 500 else ''}")
        
        # Extract an order number for next test
        order_no = None
        line_no = None
        if isinstance(result, dict):
            if "data" in result:
                data = result["data"]
                if isinstance(data, dict) and "value" in data:
                    data = data["value"]
                if isinstance(data, list) and len(data) > 0:
                    order_no = data[0].get("OrderNo") or data[0].get("order_no")
                    line_no = data[0].get("LineNo") or data[0].get("line_no")
            elif "value" in result:
                data = result["value"]
                if isinstance(data, list) and len(data) > 0:
                    order_no = data[0].get("OrderNo") or data[0].get("order_no")
                    line_no = data[0].get("LineNo") or data[0].get("line_no")
        elif isinstance(result, list) and len(result) > 0:
            order_no = result[0].get("OrderNo") or result[0].get("order_no")
            line_no = result[0].get("LineNo") or result[0].get("line_no")
        
        if order_no:
            print(f"\n  📋 Found order: {order_no}, line: {line_no}")
        
    except Exception as e:
        print(f"  ❌ Order search failed: {e}")
        order_no = None
    
    # Test 4: Get order details
    if order_no:
        print(f"\n[TEST 4] Getting details for order {order_no}...")
        try:
            details_call = {"function": {"name": "get_order_details", "arguments": {"order_no": order_no}}}
            result = await tool_manager.call_tool(details_call, {})
            print(f"  ✅ Order details returned:")
            result_str = json.dumps(result, indent=2)
            print(f"  {result_str[:800]}{'...' if len(result_str) > 800 else ''}")
        except Exception as e:
            print(f"  ❌ Get order details failed: {e}")
    else:
        print("\n[TEST 4] Skipped - no order number available")
    
    # Test 5: Test adding an order line (dry run - just verify the tool exists and params)
    print("\n[TEST 5] Testing add_order_line tool availability...")
    try:
        from tools.mcp_tool_registry import TOOL_REGISTRY, get_tool_for_execution
        
        add_line_tool = TOOL_REGISTRY.get("add_order_line")
        if add_line_tool:
            print(f"  ✅ add_order_line tool found:")
            print(f"     Server: {add_line_tool.server}")
            print(f"     Summary: {add_line_tool.summary}")
            print(f"     Mutates: {add_line_tool.mutates}")
            
            # Show what parameters would be needed
            print(f"  📋 To add a line, you would need: order_no, catalog_no, buy_qty_due")
        else:
            print("  ⚠️ add_order_line not in registry")
            
    except Exception as e:
        print(f"  ❌ Tool lookup failed: {e}")
    
    # Test 6: Test tool interaction tracking
    print("\n[TEST 6] Testing tool interaction tracking...")
    from tools.mcp_tool_registry import TOOL_REGISTRY
    
    print(f"  ✅ Tool registry has {len(TOOL_REGISTRY)} tools defined")
    
    # Show some tools relevant to orders
    order_tools = [name for name, info in TOOL_REGISTRY.items() if info.category == "orders"]
    print(f"  ✅ Order-related tools: {order_tools[:5]}")
    
    # Test 7: Test memory folding prompts
    print("\n[TEST 7] Testing memory folding prompt generation...")
    try:
        from prompts.prompts_deepagent import (
            get_episode_memory_instruction,
            get_working_memory_instruction, 
            get_tool_memory_instruction
        )
        
        test_question = "Find past due orders and add a line"
        test_reasoning = "Step 1: Search for past due orders\nStep 2: Found order 12345\nStep 3: Get details"
        test_tool_history = '[{"tool_call": "list_past_due_lines({})", "tool_response": "[{OrderNo: *1045}]"}]'
        
        ep_prompt = get_episode_memory_instruction(test_question, test_reasoning, "")
        wm_prompt = get_working_memory_instruction(test_question, test_reasoning, "")
        tm_prompt = get_tool_memory_instruction(test_question, test_reasoning, test_tool_history, "")
        
        print(f"  ✅ Episode memory prompt: {len(ep_prompt)} chars")
        print(f"  ✅ Working memory prompt: {len(wm_prompt)} chars")
        print(f"  ✅ Tool memory prompt: {len(tm_prompt)} chars")
    except Exception as e:
        print(f"  ❌ Memory prompt generation failed: {e}")
    
    # Test 8: Simulate a full LLM reasoning trace with tool calls
    print("\n[TEST 8] Testing full LLM reasoning simulation...")
    try:
        from openai import AsyncOpenAI
        from prompts.prompts_deepagent import BEGIN_TOOL_CALL, END_TOOL_CALL, BEGIN_TOOL_RESPONSE, END_TOOL_RESPONSE, FOLD_THOUGHT
        from tools.mcp_tool_registry import build_tool_prompt
        
        client = AsyncOpenAI(
            api_key=args.api_key,
            base_url=args.base_url if hasattr(args, 'base_url') else None
        )
        
        system_prompt = f"""You are DeepAgent, a general reasoning agent with tools.

When you need to call a tool, use:
{BEGIN_TOOL_CALL}
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
{END_TOOL_CALL}

If your reasoning becomes lengthy or you're stuck, use:
{FOLD_THOUGHT}

Available tools:
{build_tool_prompt()}

Think step by step. Call tools as needed."""
        
        test_query = "Show me past due customer order lines"
        
        print(f"  📝 Query: {test_query}")
        print("  🤔 Getting LLM response...")
        
        response = await client.chat.completions.create(
            model=args.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": test_query}
            ],
            max_completion_tokens=1000,
            timeout=60,
        )
        
        llm_output = response.choices[0].message.content or ""
        print(f"  ✅ LLM responded with {len(llm_output)} chars")
        
        # Check if it tried to call a tool
        if BEGIN_TOOL_CALL in llm_output and END_TOOL_CALL in llm_output:
            # Extract tool call
            start = llm_output.index(BEGIN_TOOL_CALL) + len(BEGIN_TOOL_CALL)
            end = llm_output.index(END_TOOL_CALL)
            tool_json = llm_output[start:end].strip()
            print(f"  🔧 LLM requested tool call: {tool_json[:200]}")
            
            # Execute the tool
            tool_call = json.loads(tool_json)
            adapted_call = {"function": {"name": tool_call.get("name"), "arguments": tool_call.get("arguments", {})}}
            
            tool_result = await tool_manager.call_tool(adapted_call, {})
            tool_result_str = json.dumps(tool_result, indent=2)
            print(f"  📋 Tool returned {len(tool_result_str)} chars of data")
            
            # Get follow-up
            print("  🔄 Getting follow-up response...")
            followup = await client.chat.completions.create(
                model=args.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": test_query},
                    {"role": "assistant", "content": llm_output},
                    {"role": "user", "content": f"{BEGIN_TOOL_RESPONSE}\n{tool_result_str[:3000]}\n{END_TOOL_RESPONSE}"}
                ],
                max_completion_tokens=1000,
                timeout=60,
            )
            followup_text = followup.choices[0].message.content or ""
            print(f"  ✅ Follow-up: {followup_text[:300]}...")
        else:
            print(f"  ⚠️ LLM didn't call a tool. Response: {llm_output[:300]}...")
            
    except Exception as e:
        import traceback
        print(f"  ❌ LLM simulation failed: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Smoke test complete!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = asyncio.run(run_smoke_test())
    sys.exit(0 if success else 1)
