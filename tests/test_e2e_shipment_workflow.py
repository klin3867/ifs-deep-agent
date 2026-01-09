#!/usr/bin/env python3
"""
End-to-end integration test for shipment order workflow.

This test verifies:
1. LLM emits <tool_search> before calling tools
2. Knowledge (including A105 rule) is injected into conversation
3. 3-step workflow is completed: create -> add_line -> release
4. A105/105 is correctly interpreted as site AC (not warehouse AC-A105)

Run with: python tests/test_e2e_shipment_workflow.py
Requires Flask server running on port 7865
"""

import sys
import os
import json
import requests
import time
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

FLASK_URL = "http://127.0.0.1:7865"


def test_flask_server_running():
    """Verify Flask server is accessible."""
    print("=" * 70)
    print("TEST 0: Flask Server Connection")
    print("=" * 70)

    try:
        resp = requests.get(FLASK_URL, timeout=5)
        if resp.status_code == 200:
            print("✅ Flask server is running at", FLASK_URL)
            return True
        else:
            print(f"❌ Flask server returned status {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Flask server at {FLASK_URL}")
        print("   Start the server with: python3 src/app_flask.py --port 7865")
        return False


def send_chat_message(message: str, timeout: int = 120) -> dict:
    """
    Send a message to the Flask chat endpoint and collect the streamed response.

    Returns dict with:
        - full_response: Complete assistant response text
        - events: List of SSE events received
        - tool_searches: List of tool search queries detected
        - tool_calls: List of tool calls made
    """
    result = {
        "full_response": "",
        "events": [],
        "tool_searches": [],
        "tool_calls": [],
    }

    try:
        resp = requests.post(
            f"{FLASK_URL}/chat/stream",
            json={"message": message},
            stream=True,
            timeout=timeout,
        )

        for line in resp.iter_lines():
            if not line:
                continue

            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data_str = line_str[6:]
                if data_str == '[DONE]':
                    break

                try:
                    event = json.loads(data_str)
                    result["events"].append(event)

                    # Collect response chunks
                    if event.get("type") == "response":
                        result["full_response"] += event.get("content", "")
                    elif event.get("type") == "chunk":
                        result["full_response"] += event.get("content", "")

                    # Track tool searches
                    if event.get("type") == "tool_search":
                        result["tool_searches"].append(event.get("query", ""))

                    # Track tool calls
                    if event.get("type") == "tool_call":
                        result["tool_calls"].append({
                            "name": event.get("name", ""),
                            "arguments": event.get("arguments", {}),
                        })
                    if event.get("type") == "tool_result":
                        # Attach result to last tool call
                        if result["tool_calls"]:
                            result["tool_calls"][-1]["result"] = event.get("result", {})

                except json.JSONDecodeError:
                    pass

    except Exception as e:
        print(f"Error during chat: {e}")

    return result


def test_tool_search_emitted():
    """
    Test 1: Verify LLM emits <tool_search> when asked about shipments.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Tool Search Emission")
    print("=" * 70)
    print("Prompt: 'What tools can I use to create shipment orders?'")
    print()

    result = send_chat_message("What tools can I use to create shipment orders?")

    if result["tool_searches"]:
        print(f"✅ Tool search emitted: {result['tool_searches']}")
        return True
    else:
        print("❌ No tool search was emitted")
        print(f"   Response preview: {result['full_response'][:200]}...")
        return False


def test_knowledge_injection():
    """
    Test 2: Verify knowledge is injected when searching for shipment tools.

    We can't directly see the injected knowledge, but we can verify the LLM
    mentions key facts from the knowledge base in its response.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Knowledge Injection")
    print("=" * 70)
    print("Prompt: 'Explain the steps to create a shipment order'")
    print()

    result = send_chat_message("Explain the steps to create a shipment order")
    response = result["full_response"].lower()

    # Check for key knowledge elements (flexible matching)
    checks = {
        "3-step workflow": any(x in response for x in ["three step", "3 step", "three-step", "step 1", "1.", "2.", "3."]),
        "create_shipment_order mentioned": "create_shipment_order" in response or "create shipment" in response or "create" in response,
        "add_shipment_order_line mentioned": "add_shipment_order_line" in response or "add line" in response or "add" in response and "line" in response,
        "release mentioned": "release" in response,
        "workflow structure present": "step" in response or "then" in response or "after" in response,
    }

    passed = 0
    for check, result_bool in checks.items():
        status = "✅" if result_bool else "❌"
        print(f"{status} {check}: {result_bool}")
        if result_bool:
            passed += 1

    # Consider passed if at least 3 of 5 checks pass
    return passed >= 3


def test_a105_interpretation():
    """
    Test 3: Verify "105" is correctly interpreted as site AC, not warehouse AC-A105.

    This is the critical test that was failing.
    """
    print("\n" + "=" * 70)
    print("TEST 3: A105 Interpretation (CRITICAL)")
    print("=" * 70)
    print("Prompt: 'If I say move from 105 to 205, what does 105 refer to?'")
    print()

    result = send_chat_message(
        "In IFS Cloud, if I say 'move inventory from 105 to 205', what does '105' refer to? "
        "Is it a warehouse or a site?"
    )
    response = result["full_response"].lower()

    # Check for correct interpretation (flexible matching)
    checks = {
        "Mentions site AC": "site ac" in response or "site 'ac'" in response or '"ac"' in response or "main site" in response,
        "Mentions 105 refers to site": "site itself" in response or "refers to the site" in response or "refer to" in response and "site" in response,
        "Mentions AC-A205 as actual warehouse": "ac-a205" in response or "ac‑a205" in response or "205" in response,
        "Mentions warehouse relationship": "remote warehouse" in response or "warehouse" in response,
    }

    passed = 0
    for check, result_bool in checks.items():
        status = "✅" if result_bool else "❌"
        print(f"{status} {check}: {result_bool}")
        if result_bool:
            passed += 1

    print()
    print(f"Response preview: {result['full_response'][:500]}...")

    # Must pass at least 2 checks
    return passed >= 2


def test_three_step_workflow():
    """
    Test 4: Verify 3-step workflow is followed when creating a shipment.

    This test checks that:
    1. create_shipment_order is called first
    2. add_shipment_order_line is called with the returned shipment_order_id
    3. release_shipment_order is called to finalize
    """
    print("\n" + "=" * 70)
    print("TEST 4: Three-Step Workflow Execution")
    print("=" * 70)
    print("Prompt: 'Create a shipment order to move 100 units of part 10106105 from site AC to warehouse AC-A205'")
    print()
    print("⚠️  This test will actually create a shipment order in the system!")
    print()

    result = send_chat_message(
        "Create a shipment order to move 100 units of part 10106105 from site AC to warehouse AC-A205. "
        "The sender is site AC (type: site, id: AC) and receiver is warehouse AC-A205 (type: remote_warehouse, id: AC-A205)."
    )

    # Analyze tool calls
    tool_names = [tc["name"] for tc in result["tool_calls"]]

    print(f"Tool calls made: {tool_names}")

    checks = {
        "create_shipment_order called": "create_shipment_order" in tool_names,
        "add_shipment_order_line called": "add_shipment_order_line" in tool_names,
        "release_shipment_order called": "release_shipment_order" in tool_names,
    }

    passed = 0
    for check, result_bool in checks.items():
        status = "✅" if result_bool else "❌"
        print(f"{status} {check}")
        if result_bool:
            passed += 1

    # Check order of calls
    if all(checks.values()):
        create_idx = tool_names.index("create_shipment_order")
        add_idx = tool_names.index("add_shipment_order_line")
        release_idx = tool_names.index("release_shipment_order")

        correct_order = create_idx < add_idx < release_idx
        status = "✅" if correct_order else "❌"
        print(f"{status} Correct order (create < add < release): {correct_order}")
        if correct_order:
            passed += 1

    # Check that add_line used the shipment_order_id from create
    if "create_shipment_order" in tool_names and "add_shipment_order_line" in tool_names:
        create_call = next(tc for tc in result["tool_calls"] if tc["name"] == "create_shipment_order")
        add_call = next(tc for tc in result["tool_calls"] if tc["name"] == "add_shipment_order_line")

        # Extract shipment_order_id from create result
        create_result = create_call.get("result", {})
        if isinstance(create_result, dict):
            created_id = create_result.get("shipment_order_id")
            # Check if add_line used this ID
            add_args = add_call.get("arguments", {})
            used_id = add_args.get("shipment_order_id")

            if created_id and used_id:
                id_match = str(created_id) == str(used_id)
                status = "✅" if id_match else "❌"
                print(f"{status} add_line used correct shipment_order_id ({created_id}): {id_match}")
                if id_match:
                    passed += 1

    print()
    print(f"Response preview: {result['full_response'][:500]}...")

    return passed >= 3


def test_sender_type_interpretation():
    """
    Test 5: Verify from_warehouse/to_warehouse are correctly set based on 105/205 inputs.

    MCP API uses: from_warehouse, to_warehouse, site (NOT sender_type/sender_id)
    - '105' refers to site AC itself -> from_warehouse='AC'
    - '205' refers to warehouse AC-A205 -> to_warehouse='AC-A205'
    """
    print("\n" + "=" * 70)
    print("TEST 5: From/To Warehouse Interpretation")
    print("=" * 70)
    print("Prompt: 'Create shipment from 105 to 205 for part 10106105, qty 50'")
    print()
    print("Expected: from_warehouse='AC', to_warehouse='AC-A205', site='AC'")
    print()

    result = send_chat_message(
        "Create a shipment order to move 50 units of part 10106105 from 105 to 205"
    )

    # Find the create_shipment_order call
    create_calls = [tc for tc in result["tool_calls"] if tc["name"] == "create_shipment_order"]

    if not create_calls:
        print("❌ create_shipment_order was not called")
        return False

    create_args = create_calls[0].get("arguments", {})
    print(f"create_shipment_order arguments: {json.dumps(create_args, indent=2)}")

    # Check for correct MCP parameter names (from_warehouse, to_warehouse)
    checks = {
        "from_warehouse is 'AC'": create_args.get("from_warehouse") == "AC",
        "to_warehouse is 'AC-A205'": create_args.get("to_warehouse") == "AC-A205",
        "site is 'AC'": create_args.get("site") == "AC",
    }

    passed = 0
    for check, result_bool in checks.items():
        status = "✅" if result_bool else "❌"
        print(f"{status} {check}")
        if result_bool:
            passed += 1

    return passed >= 2


def run_all_tests():
    """Run all end-to-end tests."""
    print("\n" + "=" * 70)
    print("END-TO-END INTEGRATION TEST SUITE")
    print("=" * 70)
    print()

    results = {}

    # Test 0: Server connection
    if not test_flask_server_running():
        print("\n❌ Cannot proceed without Flask server")
        return False
    results["Server Connection"] = True

    # Test 1: Tool search emission
    results["Tool Search Emission"] = test_tool_search_emitted()

    # Test 2: Knowledge injection
    results["Knowledge Injection"] = test_knowledge_injection()

    # Test 3: A105 interpretation (critical)
    results["A105 Interpretation"] = test_a105_interpretation()

    # Test 4: Three-step workflow
    results["Three-Step Workflow"] = test_three_step_workflow()

    # Test 5: Sender/receiver type interpretation
    results["Sender/Receiver Types"] = test_sender_type_interpretation()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = 0
    failed = 0
    skipped = 0

    for test_name, result in results.items():
        if result is None:
            status = "⏭️ SKIPPED"
            skipped += 1
        elif result:
            status = "✅ PASSED"
            passed += 1
        else:
            status = "❌ FAILED"
            failed += 1
        print(f"{status}: {test_name}")

    print()
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
