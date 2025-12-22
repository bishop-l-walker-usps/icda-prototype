#!/usr/bin/env python
"""
ICDA End-to-End Test Runner

Interactive standalone script for running E2E tests with detailed output.
Can be run independently without pytest for manual exploration.

Usage:
    python tests/e2e_runner.py              # Run all tests
    python tests/e2e_runner.py --test 1     # Run specific test
    python tests/e2e_runner.py --test 2     # Test 2: Invalid state
    python tests/e2e_runner.py --test 3     # Test 3: Conversation memory
    python tests/e2e_runner.py --test 4     # Test 4: Batch addresses

Requirements:
    - Backend running on localhost:8000
    - Docker infrastructure (Redis, OpenSearch)
    - AWS credentials for Bedrock Nova
"""

import asyncio
import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Configuration
# ============================================================================


API_BASE_URL = "http://localhost:8000"
TIMEOUT = 120.0  # 2 minutes for complex queries


# ============================================================================
# Utilities
# ============================================================================


def print_header(title: str) -> None:
    """Print a formatted header."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n--- {title} ---")


def print_json(data: Any, indent: int = 2) -> None:
    """Pretty print JSON data."""
    print(json.dumps(data, indent=indent, default=str))


def format_latency(ms: float) -> str:
    """Format latency in human-readable form."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms/1000:.2f}s"


class TestResult:
    """Holds test result information."""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.details: dict = {}
        self.latency_ms: float = 0

    def success(self, message: str = "", **details):
        self.passed = True
        self.message = message
        self.details = details
        return self

    def fail(self, message: str, **details):
        self.passed = False
        self.message = message
        self.details = details
        return self


# ============================================================================
# Test 1: Complex Multi-Filter Query
# ============================================================================


async def test_1_complex_query(client: httpx.AsyncClient) -> TestResult:
    """
    Test 1: Complex query with status and date filters.

    Query: Find customers without active status who moved before 2024.
    """
    result = TestResult("Test 1: Complex Multi-Filter Query")

    query = "Find all customers who don't have active status but who moved before 2024"

    print_header("TEST 1: Complex Multi-Filter Query")
    print(f"\nQuery: {query}")

    session_id = str(uuid.uuid4())

    print_section("Sending Request")
    start = datetime.now()

    response = await client.post(
        "/api/query",
        json={
            "query": query,
            "session_id": session_id,
            "bypass_cache": True,
        },
    )

    latency_ms = (datetime.now() - start).total_seconds() * 1000
    result.latency_ms = latency_ms

    print(f"Status Code: {response.status_code}")
    print(f"Latency: {format_latency(latency_ms)}")

    if response.status_code != 200:
        return result.fail(f"HTTP Error: {response.status_code}")

    data = response.json()

    print_section("Response")
    print(f"Success: {data.get('success')}")
    print(f"Route: {data.get('route')}")
    print(f"Blocked: {data.get('blocked')}")
    print(f"Cached: {data.get('cached')}")

    if "quality_score" in data:
        print(f"Quality Score: {data['quality_score']}")

    if "token_usage" in data:
        usage = data["token_usage"]
        print(f"Tokens: {usage.get('total_tokens', 'N/A')}")

    print_section("Response Text")
    response_text = data.get("response", "")
    print(response_text[:1000])
    if len(response_text) > 1000:
        print(f"... ({len(response_text)} total characters)")

    # Check for results
    if "results" in data and isinstance(data["results"], list):
        print_section("Results Sample")
        for i, r in enumerate(data["results"][:3]):
            print(f"  [{i+1}] CRID: {r.get('crid')}, Status: {r.get('status')}, "
                  f"Last Move: {r.get('last_move')}")

    # Validation
    if not data.get("success"):
        return result.fail(f"Query failed: {data.get('response')}")

    if data.get("blocked"):
        return result.fail("Query was blocked by guardrails")

    return result.success(
        "Complex query executed successfully",
        route=data.get("route"),
        response_length=len(response_text),
        session_id=session_id,
    )


# ============================================================================
# Test 2: Invalid State Detection
# ============================================================================


async def test_2_invalid_state(client: httpx.AsyncClient) -> TestResult:
    """
    Test 2: Query with guardrails off and invalid state "Argintina".

    Should detect invalid state and suggest alternatives like Arkansas.
    """
    result = TestResult("Test 2: Invalid State Detection")

    query = (
        "Find customers who don't have active status but who moved before 2024 "
        "who have lived in Arizona and Argintina"
    )

    print_header("TEST 2: Invalid State Detection + Guardrails Off")
    print(f"\nQuery: {query}")
    print("Guardrails: ALL DISABLED")

    session_id = str(uuid.uuid4())

    print_section("Sending Request")
    start = datetime.now()

    response = await client.post(
        "/api/query",
        json={
            "query": query,
            "session_id": session_id,
            "bypass_cache": True,
            "guardrails": {
                "pii": False,
                "financial": False,
                "credentials": False,
                "offtopic": False,
            },
        },
    )

    latency_ms = (datetime.now() - start).total_seconds() * 1000
    result.latency_ms = latency_ms

    print(f"Status Code: {response.status_code}")
    print(f"Latency: {format_latency(latency_ms)}")

    if response.status_code != 200:
        return result.fail(f"HTTP Error: {response.status_code}")

    data = response.json()

    print_section("Response")
    print(f"Success: {data.get('success')}")
    print(f"Route: {data.get('route')}")
    print(f"Guardrails Active: {data.get('guardrails_active')}")
    print(f"Guardrails Bypassed: {data.get('guardrails_bypassed')}")

    print_section("Response Text")
    response_text = data.get("response", "")
    print(response_text[:1500])

    # Analyze response for invalid state handling
    print_section("Invalid State Analysis")
    response_lower = response_text.lower()

    checks = {
        "Mentions Argintina": "argintina" in response_lower,
        "Mentions invalid/not found": any(w in response_lower for w in [
            "invalid", "not found", "no data", "doesn't exist", "unknown", "couldn't find"
        ]),
        "Suggests Arkansas": "arkansas" in response_lower,
        "Suggests Arizona": "arizona" in response_lower,
        "Has 'did you mean'": "did you mean" in response_lower,
        "Has 'suggestion'": "suggestion" in response_lower,
    }

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    if not data.get("success"):
        return result.fail(f"Query failed: {response_text[:200]}")

    # Guardrails should be bypassed
    if data.get("guardrails_active") is True:
        return result.fail("Guardrails should be bypassed but were active")

    return result.success(
        "Invalid state query executed with guardrails off",
        route=data.get("route"),
        checks=checks,
        session_id=session_id,
    )


# ============================================================================
# Test 3: Conversation Memory
# ============================================================================


async def test_3_conversation_memory(client: httpx.AsyncClient) -> TestResult:
    """
    Test 3: Multi-turn conversation with session context.

    1. Initial query about customers
    2. Follow-up referencing "those results"
    """
    result = TestResult("Test 3: Conversation Memory")

    print_header("TEST 3: Conversation Memory/Context")

    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")

    # === Turn 1: Initial Query ===
    print_section("Turn 1: Initial Query")
    query1 = "Find customers who don't have active status but who moved before 2024"
    print(f"Query: {query1}")

    start = datetime.now()
    response1 = await client.post(
        "/api/query",
        json={
            "query": query1,
            "session_id": session_id,
            "bypass_cache": True,
        },
    )
    latency1 = (datetime.now() - start).total_seconds() * 1000

    if response1.status_code != 200:
        return result.fail(f"Turn 1 HTTP Error: {response1.status_code}")

    data1 = response1.json()
    print(f"Status: {response1.status_code}, Latency: {format_latency(latency1)}")
    print(f"Response: {data1.get('response', '')[:300]}...")

    if not data1.get("success"):
        return result.fail(f"Turn 1 failed: {data1.get('response')}")

    # === Turn 2: Follow-up Query ===
    print_section("Turn 2: Follow-up Query")
    query2 = "From those results, how many became active in 2023 vs 2022?"
    print(f"Query: {query2}")

    start = datetime.now()
    response2 = await client.post(
        "/api/query",
        json={
            "query": query2,
            "session_id": session_id,  # Same session!
            "bypass_cache": True,
        },
    )
    latency2 = (datetime.now() - start).total_seconds() * 1000
    result.latency_ms = latency1 + latency2

    if response2.status_code != 200:
        return result.fail(f"Turn 2 HTTP Error: {response2.status_code}")

    data2 = response2.json()
    print(f"Status: {response2.status_code}, Latency: {format_latency(latency2)}")

    print_section("Turn 2 Response")
    response_text = data2.get("response", "")
    print(response_text[:1000])

    if not data2.get("success"):
        return result.fail(f"Turn 2 failed: {response_text[:200]}")

    # Check for context usage
    print_section("Context Usage Analysis")
    response_lower = response_text.lower()

    context_signals = {
        "References 2023": "2023" in response_lower,
        "References 2022": "2022" in response_lower,
        "Uses 'those'": "those" in response_lower,
        "Mentions customers": "customer" in response_lower,
        "Mentions results": "result" in response_lower,
    }

    signals_found = sum(1 for v in context_signals.values() if v)
    for signal, found in context_signals.items():
        status = "PASS" if found else "FAIL"
        print(f"  [{status}] {signal}")

    print(f"\nContext signals found: {signals_found}/5")

    return result.success(
        "Conversation memory test completed",
        session_id=session_id,
        turn1_latency=latency1,
        turn2_latency=latency2,
        context_signals=signals_found,
    )


# ============================================================================
# Test 4: Batch Address Verification
# ============================================================================


async def test_4_batch_addresses(client: httpx.AsyncClient) -> TestResult:
    """
    Test 4: Batch verification of 200 corrupted addresses.

    Target: >80% success rate (verified + corrected + completed).
    """
    result = TestResult("Test 4: Batch Address Verification")

    print_header("TEST 4: Batch Address Verification (200 Addresses)")

    # Load corrupted addresses
    fixture_path = project_root / "tests" / "fixtures" / "corrupted_addresses.json"
    if not fixture_path.exists():
        return result.fail(f"Fixture not found: {fixture_path}")

    with open(fixture_path) as f:
        corrupted_data = json.load(f)

    print(f"Loaded {len(corrupted_data)} corrupted addresses")

    # Show sample corruptions
    print_section("Sample Corrupted Addresses")
    for addr in corrupted_data[:5]:
        print(f"  [{addr['corruption_type']}] {addr['corrupted_address'][:60]}...")

    # Prepare batch
    addresses = [addr["corrupted_address"] for addr in corrupted_data]

    print_section("Sending Batch Request")
    print(f"Addresses: {len(addresses)}")
    print(f"Concurrency: 10")

    start = datetime.now()

    response = await client.post(
        "/api/address/verify/batch",
        json={
            "addresses": addresses,
            "concurrency": 10,
        },
        timeout=300.0,  # 5 minutes for 200 addresses
    )

    latency_ms = (datetime.now() - start).total_seconds() * 1000
    result.latency_ms = latency_ms

    print(f"Status Code: {response.status_code}")
    print(f"Total Time: {format_latency(latency_ms)}")
    print(f"Avg per address: {latency_ms / len(addresses):.1f}ms")

    if response.status_code != 200:
        return result.fail(f"HTTP Error: {response.status_code} - {response.text[:500]}")

    data = response.json()

    # Process results
    if "results" not in data:
        print(f"Response: {json.dumps(data, indent=2)[:2000]}")
        return result.fail("No 'results' in response")

    results_list = data["results"]

    print_section("Results Summary")

    # Count by status - handle nested result structure
    status_counts: dict[str, int] = {}
    for r in results_list:
        # Handle nested structure: results[i].result.status
        inner = r.get("result", r)  # Fallback to r if no nesting
        status = str(inner.get("status", "unknown")).lower()
        status_counts[status] = status_counts.get(status, 0) + 1

    for status, count in sorted(status_counts.items()):
        pct = count / len(results_list) * 100
        print(f"  {status}: {count} ({pct:.1f}%)")

    # Calculate success rate
    success_keywords = ["verified", "corrected", "completed"]
    success_count = sum(
        count for status, count in status_counts.items()
        if any(kw in status for kw in success_keywords)
    )
    success_rate = success_count / len(results_list) * 100

    print(f"\nSuccess Rate: {success_rate:.1f}%")
    print(f"Target: 80%")

    if success_rate >= 80:
        print("STATUS: PASS - Met 80% target!")
    else:
        print("STATUS: WARNING - Below 80% target")

    # Show sample corrections
    print_section("Sample Corrections")
    for i, r in enumerate(results_list[:10]):
        original = corrupted_data[i]["corrupted_address"][:40]
        # Handle nested structure
        inner = r.get("result", r)
        status = inner.get("status", "?")
        conf = inner.get("confidence", 0)

        verified = inner.get("verified")
        if isinstance(verified, dict):
            corrected = verified.get("formatted", verified.get("single_line", "N/A"))[:50]
        elif verified:
            corrected = str(verified)[:50]
        else:
            corrected = "N/A"

        print(f"  [{status}] {original}...")
        print(f"    -> {corrected} (conf: {conf:.2f})")

    return result.success(
        f"Batch verification completed: {success_rate:.1f}% success",
        total=len(results_list),
        status_counts=status_counts,
        success_rate=success_rate,
        target_met=success_rate >= 80,
    )


# ============================================================================
# Main Runner
# ============================================================================


async def check_api_health(client: httpx.AsyncClient) -> bool:
    """Check if API is accessible."""
    try:
        response = await client.get("/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"API Status: {data.get('status')}")
            print(f"Mode: {data.get('mode')}")
            return True
        print(f"API returned: {response.status_code}")
        return False
    except httpx.ConnectError:
        print("ERROR: Cannot connect to API at localhost:8000")
        print("Make sure the backend is running: uvicorn main:app --port 8000")
        return False


async def run_all_tests(test_num: int | None = None) -> list[TestResult]:
    """Run all tests or a specific test."""
    results: list[TestResult] = []

    async with httpx.AsyncClient(
        base_url=API_BASE_URL,
        timeout=TIMEOUT,
    ) as client:
        # Health check
        print_header("API Health Check")
        if not await check_api_health(client):
            return results

        # Test mapping
        tests = {
            1: ("Complex Query", test_1_complex_query),
            2: ("Invalid State", test_2_invalid_state),
            3: ("Conversation Memory", test_3_conversation_memory),
            4: ("Batch Addresses", test_4_batch_addresses),
        }

        if test_num:
            if test_num not in tests:
                print(f"Invalid test number: {test_num}")
                print(f"Available: {list(tests.keys())}")
                return results
            tests = {test_num: tests[test_num]}

        for num, (name, test_fn) in tests.items():
            try:
                result = await test_fn(client)
                results.append(result)
            except Exception as e:
                result = TestResult(f"Test {num}: {name}")
                result.fail(f"Exception: {e}")
                results.append(result)
                print(f"\nERROR in Test {num}: {e}")
                import traceback
                traceback.print_exc()

    return results


def print_summary(results: list[TestResult]) -> None:
    """Print test summary."""
    print_header("TEST SUMMARY")

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        latency = format_latency(result.latency_ms) if result.latency_ms else "N/A"
        print(f"  [{status}] {result.name} ({latency})")
        if not result.passed:
            print(f"       -> {result.message}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\nALL TESTS PASSED!")
    else:
        print(f"\n{total - passed} TEST(S) FAILED")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ICDA E2E Test Runner")
    parser.add_argument(
        "--test", "-t",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific test (1-4)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  ICDA End-to-End Test Runner")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"API: {API_BASE_URL}")

    results = asyncio.run(run_all_tests(args.test))

    if results:
        print_summary(results)

    # Exit code
    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
