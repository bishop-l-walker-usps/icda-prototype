"""Standalone address validation test - NO AWS required.

Loads and tests all 100 addresses from test_addresses_validation.json.

Author: Bishop Walker
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, '.')

from icda.address_normalizer import AddressNormalizer
from icda.address_validator_engine import AddressValidatorEngine, ValidationMode


def load_test_addresses():
    """Load all test addresses from JSON file."""
    json_path = Path(__file__).parent / "test_addresses_validation.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_expected_correction(entry: dict) -> dict:
    """Extract expected correction details from JSON entry."""
    expected = {}
    error_type = entry["error_type"]

    # Map error types to their expected correction fields
    if "correct_city" in entry:
        expected["city"] = entry["correct_city"]
    if "correct_street" in entry:
        expected["street"] = entry["correct_street"]
    if "correct_zip" in entry:
        expected["zip"] = entry["correct_zip"]
    if "original_zip" in entry:
        expected["original_zip"] = entry["original_zip"]
    if "correct_address" in entry:
        expected["full_address"] = entry["correct_address"]
    if "correct_state" in entry:
        expected["state"] = entry["correct_state"]
    if "correct_format" in entry:
        expected["format"] = entry["correct_format"]
    if "note" in entry:
        expected["note"] = entry["note"]

    return expected


def check_correction_applied(result, entry: dict, expected: dict) -> tuple[bool, str]:
    """Check if the validator correctly identified/fixed the problem."""
    error_type = entry["error_type"]
    standardized = result.standardized.upper() if result.standardized else ""
    issues_text = " ".join([i.message.lower() for i in result.issues]) if result.issues else ""

    # Check based on error type
    if error_type == "misspelled_city":
        correct_city = expected.get("city", "").upper()
        if correct_city and correct_city in standardized:
            return True, f"City corrected to {correct_city}"
        return False, f"City NOT corrected (expected: {expected.get('city')})"

    elif error_type == "misspelled_street":
        correct_street = expected.get("street", "").upper()
        if correct_street:
            # Extract just the street name part for comparison
            correct_parts = correct_street.split()
            if len(correct_parts) > 1:
                street_name = correct_parts[1]  # e.g., "ASPEN" from "5075 ASPEN LN"
                if street_name in standardized:
                    return True, f"Street corrected"
        return False, f"Street NOT corrected (expected: {expected.get('street')})"

    elif error_type == "missing_zip":
        original_zip = expected.get("original_zip", "")
        if original_zip and original_zip in standardized:
            return True, f"ZIP completed to {original_zip}"
        if "missing zip" in issues_text or "zip" in issues_text:
            return False, f"ZIP missing flagged but not completed (need: {original_zip})"
        return False, f"ZIP NOT completed (expected: {original_zip})"

    elif error_type == "wrong_zip_for_city":
        correct_zip = expected.get("zip", "")
        if correct_zip and correct_zip in standardized:
            return True, f"ZIP corrected to {correct_zip}"
        return False, f"ZIP NOT corrected (expected: {correct_zip})"

    elif error_type == "transposed_zip_digits":
        correct_zip = expected.get("zip", "")
        if correct_zip and correct_zip in standardized:
            return True, f"ZIP corrected to {correct_zip}"
        return False, f"Transposed ZIP NOT corrected (expected: {correct_zip})"

    elif error_type == "wrong_state":
        correct_state = expected.get("state", "")
        if correct_state and f", {correct_state} " in standardized:
            return True, f"State corrected to {correct_state}"
        return False, f"State NOT corrected (expected: {correct_state})"

    elif error_type == "missing_street_number":
        correct_addr = expected.get("full_address", "")
        if correct_addr:
            # Check if a street number was added
            parts = correct_addr.split()
            if parts and parts[0].isdigit():
                if parts[0] in standardized:
                    return True, f"Street number added: {parts[0]}"
        return False, f"Street number NOT added (expected: {correct_addr})"

    elif error_type == "missing_unit_number":
        correct_addr = expected.get("full_address", "")
        if "unit" in issues_text or "apt" in issues_text or "secondary" in issues_text:
            return False, f"Unit flagged but not added (expected: {correct_addr})"
        return False, f"Missing unit NOT detected (expected: {correct_addr})"

    elif error_type == "extra_spaces_formatting":
        # Check if extra spaces were normalized
        if "  " not in standardized:
            return True, "Extra spaces normalized"
        return False, "Extra spaces NOT normalized"

    elif error_type == "truncated_address":
        correct_addr = expected.get("full_address", "")
        if "missing" in issues_text or "incomplete" in issues_text:
            return False, f"Truncation flagged (expected: {correct_addr})"
        return False, f"Truncation NOT detected (expected: {correct_addr})"

    elif error_type == "mixed_case_issues":
        # Check if properly capitalized
        if standardized and standardized == standardized.upper():
            return True, "Case normalized"
        return False, "Case NOT normalized"

    return False, "Unknown error type"


def test_addresses():
    """Test all 100 addresses from the JSON file."""
    engine = AddressValidatorEngine()

    # Load test data from JSON
    test_data = load_test_addresses()
    addresses = test_data["addresses"]

    print("=" * 70)
    print("ADDRESS VALIDATION TEST (No AWS Required)")
    print(f"Testing {len(addresses)} addresses from test_addresses_validation.json")
    print("=" * 70)

    # Track statistics by error type
    stats = {
        "total": len(addresses),
        "validated": 0,
        "corrected": 0,
        "correction_success": 0,
        "by_error_type": {}
    }

    for entry in addresses:
        addr_id = entry["id"]
        address = entry["address"]
        city = entry["city"]
        state = entry["state"]
        zip_code = entry["zip"]
        error_type = entry["error_type"]

        # Get expected correction
        expected = get_expected_correction(entry)

        # Build full address string
        full_address = f"{address}, {city}, {state}"
        if zip_code:
            full_address += f" {zip_code}"

        print(f"\n[{addr_id:3d}] Input: {full_address}")
        print(f"      Error Type: {error_type}")

        # Show what the expected correction should be
        if expected:
            exp_details = []
            if "city" in expected:
                exp_details.append(f"city={expected['city']}")
            if "street" in expected:
                exp_details.append(f"street={expected['street']}")
            if "zip" in expected:
                exp_details.append(f"zip={expected['zip']}")
            if "original_zip" in expected:
                exp_details.append(f"missing_zip={expected['original_zip']}")
            if "state" in expected:
                exp_details.append(f"state={expected['state']}")
            if "full_address" in expected:
                exp_details.append(f"should_be={expected['full_address']}")
            if "format" in expected:
                exp_details.append(f"format={expected['format']}")
            if "note" in expected:
                exp_details.append(f"note={expected['note']}")
            print(f"      Expected: {', '.join(exp_details)}")

        print("-" * 70)

        result = engine.validate(full_address, ValidationMode.CORRECT)

        # Check if correction was properly applied
        correction_ok, correction_msg = check_correction_applied(result, entry, expected)

        # Track stats
        if result.is_valid:
            stats["validated"] += 1
        if result.corrections_applied:
            stats["corrected"] += 1
        if correction_ok:
            stats["correction_success"] += 1

        if error_type not in stats["by_error_type"]:
            stats["by_error_type"][error_type] = {
                "total": 0, "validated": 0, "corrected": 0, "correction_success": 0
            }
        stats["by_error_type"][error_type]["total"] += 1
        if result.is_valid:
            stats["by_error_type"][error_type]["validated"] += 1
        if result.corrections_applied:
            stats["by_error_type"][error_type]["corrected"] += 1
        if correction_ok:
            stats["by_error_type"][error_type]["correction_success"] += 1

        print(f"Valid: {result.is_valid} | Deliverable: {result.is_deliverable}")
        print(f"Confidence: {result.overall_confidence:.1%}")
        print(f"Status: {result.status.value}")
        print(f"Quality: {result.quality.value}")

        if result.is_puerto_rico:
            print(f"PR Address: Yes | URB Status: {result.urbanization_status}")

        if result.corrections_applied:
            print(f"Corrections Applied: {result.corrections_applied}")

        if result.completions_applied:
            print(f"Completions Applied: {result.completions_applied}")

        if result.standardized:
            print(f"Standardized: {result.standardized}")

        # Show correction check result with pass/fail indicator
        status_icon = "PASS" if correction_ok else "FAIL"
        print(f"Correction Check: [{status_icon}] {correction_msg}")

        if result.issues:
            print("Issues Detected:")
            for issue in result.issues[:5]:
                print(f"  [{issue.severity}] {issue.message}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total addresses tested:    {stats['total']}")
    print(f"Validated successfully:    {stats['validated']} ({stats['validated']/stats['total']*100:.1f}%)")
    print(f"Corrections applied:       {stats['corrected']} ({stats['corrected']/stats['total']*100:.1f}%)")
    print(f"Corrections successful:    {stats['correction_success']} ({stats['correction_success']/stats['total']*100:.1f}%)")

    print("\n" + "=" * 70)
    print("BY ERROR TYPE (Correction Success)")
    print("=" * 70)
    print(f"{'Error Type':<26} {'Valid':>8} {'Corrected':>10} {'Success':>10}")
    print("-" * 70)
    for error_type, counts in sorted(stats["by_error_type"].items()):
        valid_pct = counts["validated"] / counts["total"] * 100 if counts["total"] > 0 else 0
        success_pct = counts["correction_success"] / counts["total"] * 100 if counts["total"] > 0 else 0
        print(f"  {error_type:<24} {counts['validated']:2d}/{counts['total']:2d} ({valid_pct:5.1f}%)  "
              f"{counts['corrected']:2d}/{counts['total']:2d}       "
              f"{counts['correction_success']:2d}/{counts['total']:2d} ({success_pct:5.1f}%)")

    # Show failures summary
    failed_types = [et for et, c in stats["by_error_type"].items() if c["correction_success"] < c["total"]]
    if failed_types:
        print("\n" + "=" * 70)
        print("NEEDS IMPROVEMENT (Correction not fully working)")
        print("=" * 70)
        for error_type in sorted(failed_types):
            counts = stats["by_error_type"][error_type]
            failed = counts["total"] - counts["correction_success"]
            print(f"  {error_type}: {failed} failures out of {counts['total']}")


if __name__ == "__main__":
    test_addresses()
