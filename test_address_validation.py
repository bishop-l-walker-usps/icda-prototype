"""Standalone address validation test - NO AWS required.

Author: Bishop Walker
"""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

from icda.address_validator_engine import AddressValidatorEngine, ValidationMode

# Load the fixture relative to this file so it works from any working directory.
_THIS_DIR = Path(__file__).resolve().parent
DATA_FILE = _THIS_DIR / "test_addresses_validation.json"


def load_test_cases() -> list[dict[str, Any]]:
    """Load test cases from the JSON fixture.

    Supports either schema:
      - {"test_cases": [{"input": "..."}, ...]}
      - {"addresses": [{"address": "...", "city": "...", ...}, ...]}

    Returns:
        list[dict[str, Any]]: List of test case dictionaries.

    Raises:
        FileNotFoundError: If the fixture file can't be found.
        ValueError: If the JSON can't be parsed.
        KeyError: If the JSON schema isn't recognized.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Test data file not found: {DATA_FILE}")

    try:
        payload: dict[str, Any] = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {DATA_FILE}: {exc}") from exc

    if isinstance(payload.get("test_cases"), list):
        return payload["test_cases"]

    if isinstance(payload.get("addresses"), list):
        normalized: list[dict[str, Any]] = []
        for row in payload["addresses"]:
            address = str(row.get("address", "")).strip()
            city = str(row.get("city", "")).strip()
            state = str(row.get("state", "")).strip()
            zip_code = str(row.get("zip", "")).strip()

            parts = [p for p in [address, city, state, zip_code] if p]
            normalized.append(
                {
                    "input": ", ".join(parts),
                    "description": row.get("error_type"),
                    "_raw": row,
                }
            )
        return normalized

    raise KeyError(
        f"Unexpected JSON schema in {DATA_FILE}. Expected 'test_cases' or 'addresses'."
    )


def test_addresses_from_file() -> None:
    """Run the address validator across all fixture addresses and print results."""
    engine = AddressValidatorEngine()
    test_cases = load_test_cases()

    print("=" * 70)
    print("ADDRESS VALIDATION TEST (Fixture File)")
    print("=" * 70)

    for case in test_cases:
        addr = case["input"]
        desc = case.get("description", "")

        print(f"\nInput: {addr}")
        if desc:
            print(f"Description: {desc}")
        print("-" * 50)

        result = engine.validate(addr, ValidationMode.CORRECT)

        print(f"Valid: {result.is_valid} | Deliverable: {result.is_deliverable}")
        print(f"Confidence: {result.overall_confidence:.1%}")
        print(f"Status: {result.status.value}")
        print(f"Quality: {result.quality.value}")


def test_addresses() -> None:
    """Quick smoke test using the same fixture loader."""
    engine = AddressValidatorEngine()
    test_cases = load_test_cases()

    print("=" * 70)
    print("ADDRESS VALIDATION TEST (No AWS Required)")
    print("=" * 70)

    for case in test_cases:
        addr = case["input"]

        print(f"\ninput: {addr}")
        print("-" * 50)

        result = engine.validate(addr, ValidationMode.CORRECT)

        print(f"Valid: {result.is_valid} | Deliverable: {result.is_deliverable}")
        print(f"Confidence: {result.overall_confidence:.1%}")
        print(f"Status: {result.status.value}")
        print(f"Quality: {result.quality.value}")

        if result.is_puerto_rico:
            print(f"PR Address: Yes | URB Status: {result.urbanization_status}")

        if result.corrections_applied:
            print(f"Corrections: {result.corrections_applied}")

        if result.completions_applied:
            print(f"Completions: {result.completions_applied}")

        if result.standardized:
            print(f"Standardized: {result.standardized}")

        if result.issues:
            print("Issues:")
            for issue in result.issues[:3]:
                print(f"  [{issue.severity}] {issue.message}")


if __name__ == "__main__":
    test_addresses()
    test_addresses_from_file()
