"""Enhanced Address Validation Test Runner.

Runs the 6 address validation enforcer agents against address data files.
Supports flexible file input (JSON/CSV), auto-field mapping, and
presentation-quality output for demos.

Usage:
    python test_address_validation.py                           # Default JSON file
    python test_address_validation.py batch_addresses.json      # Custom JSON
    python test_address_validation.py addresses.csv             # CSV file
    python test_address_validation.py data.json --format=json   # JSON output
    python test_address_validation.py data.json --format=csv    # CSV output

Author: Bishop Walker
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, ".")

from icda.address_normalizer import AddressNormalizer
from icda.address_validator_engine import AddressValidatorEngine, ValidationMode
from icda.agents.enforcers.address import (
    AddressEnforcerCoordinator,
    AddressInput,
    BatchConfiguration,
    BatchOrchestratorAgent,
    parse_addresses_from_file,
)


def load_file(file_path: str) -> tuple[list[dict], str]:
    """Load addresses from JSON or CSV file.

    Args:
        file_path: Path to the file.

    Returns:
        Tuple of (data, format) where format is 'json' or 'csv'.
    """
    path = Path(file_path)

    if not path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, "json"

    elif suffix == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        return data, "csv"

    else:
        # Try JSON first
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data, "json"
        except json.JSONDecodeError:
            # Try CSV
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = list(reader)
            return data, "csv"


def get_expected_correction(entry: dict) -> dict:
    """Extract expected correction details from entry."""
    expected = {}

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
    error_type = entry.get("error_type", "")
    standardized = result.standardized.upper() if result.standardized else ""
    issues_text = " ".join([i.message.lower() for i in result.issues]) if result.issues else ""

    if error_type == "misspelled_city":
        correct_city = expected.get("city", "").upper()
        if correct_city and correct_city in standardized:
            return True, f"City corrected to {correct_city}"
        return False, f"City NOT corrected (expected: {expected.get('city')})"

    elif error_type == "misspelled_street":
        correct_street = expected.get("street", "").upper()
        if correct_street:
            correct_parts = correct_street.split()
            if len(correct_parts) > 1:
                street_name = correct_parts[1]
                if street_name in standardized:
                    return True, "Street corrected"
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
        if "  " not in standardized:
            return True, "Extra spaces normalized"
        return False, "Extra spaces NOT normalized"

    elif error_type == "truncated_address":
        correct_addr = expected.get("full_address", "")
        if "missing" in issues_text or "incomplete" in issues_text:
            return False, f"Truncation flagged (expected: {correct_addr})"
        return False, f"Truncation NOT detected (expected: {correct_addr})"

    elif error_type == "mixed_case_issues":
        if standardized and standardized == standardized.upper():
            return True, "Case normalized"
        return False, "Case NOT normalized"

    return False, "Unknown error type"


async def run_with_enforcers(
    file_path: str,
    output_format: str = "presentation",
    verbose: bool = False,
) -> None:
    """Run address validation with all 6 enforcer agents.

    Args:
        file_path: Path to the address file.
        output_format: Output format (presentation, json, csv).
        verbose: Show detailed per-address output.
    """
    print("=" * 80)
    print("ADDRESS VALIDATION WITH ENFORCER AGENTS")
    print("=" * 80)
    print(f"File: {file_path}")
    print(f"Output format: {output_format}")
    print("")

    # Load file
    data, file_format = load_file(file_path)
    print(f"Loaded {file_format.upper()} file")

    # Parse addresses
    addresses = parse_addresses_from_file(file_path, data)
    print(f"Parsed {len(addresses)} addresses")
    print("")

    # Create validator and orchestrator
    engine = AddressValidatorEngine()
    config = BatchConfiguration(
        concurrency=5,
        enable_normalization_enforcer=True,
        enable_completion_enforcer=True,
        enable_correction_enforcer=True,
        enable_match_confidence_enforcer=True,
        enable_report_generation=True,
    )

    orchestrator = BatchOrchestratorAgent(config=config)

    # Process batch
    print("Processing addresses with 6 enforcer agents...")
    print("-" * 80)

    start_time = time.time()
    report = await orchestrator.process_batch(addresses, validator=engine)
    elapsed_ms = (time.time() - start_time) * 1000

    # Output based on format
    if output_format == "json":
        output = orchestrator.to_json()
        print(json.dumps(output, indent=2))

    elif output_format == "csv":
        rows = orchestrator.to_csv_rows()
        if rows:
            writer = csv.DictWriter(sys.stdout, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    else:  # presentation
        print("")
        print(orchestrator.format_report(width=80))

    print("")
    print(f"Total processing time: {elapsed_ms:.0f} ms")


def run_basic_validation(file_path: str, verbose: bool = True) -> None:
    """Run basic address validation (legacy mode without enforcers).

    Args:
        file_path: Path to the address file.
        verbose: Show detailed per-address output.
    """
    engine = AddressValidatorEngine()

    # Load test data
    data, file_format = load_file(file_path)

    # Handle different data formats
    if isinstance(data, dict) and "addresses" in data:
        addresses = data["addresses"]
    elif isinstance(data, list):
        addresses = data
    else:
        addresses = [data]

    print("=" * 80)
    print("ADDRESS VALIDATION TEST (Basic Mode)")
    print(f"Testing {len(addresses)} addresses from {file_path}")
    print("=" * 80)

    # Track statistics by error type
    stats = {
        "total": len(addresses),
        "validated": 0,
        "corrected": 0,
        "correction_success": 0,
        "by_error_type": {},
    }

    for entry in addresses:
        addr_id = entry.get("id", 0)
        address = entry.get("address", "")
        city = entry.get("city", "")
        state = entry.get("state", "")
        zip_code = entry.get("zip", "")
        error_type = entry.get("error_type", "")

        # Get expected correction
        expected = get_expected_correction(entry)

        # Build full address string
        full_address = f"{address}, {city}, {state}"
        if zip_code:
            full_address += f" {zip_code}"

        if verbose:
            print(f"\n[{addr_id:3d}] Input: {full_address}")
            print(f"      Error Type: {error_type}")

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
                print(f"      Expected: {', '.join(exp_details)}")

            print("-" * 80)

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
                "total": 0,
                "validated": 0,
                "corrected": 0,
                "correction_success": 0,
            }
        stats["by_error_type"][error_type]["total"] += 1
        if result.is_valid:
            stats["by_error_type"][error_type]["validated"] += 1
        if result.corrections_applied:
            stats["by_error_type"][error_type]["corrected"] += 1
        if correction_ok:
            stats["by_error_type"][error_type]["correction_success"] += 1

        if verbose:
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

            status_icon = "PASS" if correction_ok else "FAIL"
            print(f"Correction Check: [{status_icon}] {correction_msg}")

            if result.issues:
                print("Issues Detected:")
                for issue in result.issues[:5]:
                    print(f"  [{issue.severity}] {issue.message}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total addresses tested:    {stats['total']}")
    print(
        f"Validated successfully:    {stats['validated']} "
        f"({stats['validated']/stats['total']*100:.1f}%)"
    )
    print(
        f"Corrections applied:       {stats['corrected']} "
        f"({stats['corrected']/stats['total']*100:.1f}%)"
    )
    print(
        f"Corrections successful:    {stats['correction_success']} "
        f"({stats['correction_success']/stats['total']*100:.1f}%)"
    )

    print("\n" + "=" * 80)
    print("BY ERROR TYPE (Correction Success)")
    print("=" * 80)
    print(f"{'Error Type':<26} {'Valid':>8} {'Corrected':>10} {'Success':>10}")
    print("-" * 80)
    for error_type, counts in sorted(stats["by_error_type"].items()):
        valid_pct = counts["validated"] / counts["total"] * 100 if counts["total"] > 0 else 0
        success_pct = (
            counts["correction_success"] / counts["total"] * 100
            if counts["total"] > 0
            else 0
        )
        print(
            f"  {error_type:<24} {counts['validated']:2d}/{counts['total']:2d} ({valid_pct:5.1f}%)  "
            f"{counts['corrected']:2d}/{counts['total']:2d}       "
            f"{counts['correction_success']:2d}/{counts['total']:2d} ({success_pct:5.1f}%)"
        )

    # Show failures summary
    failed_types = [
        et
        for et, c in stats["by_error_type"].items()
        if c["correction_success"] < c["total"]
    ]
    if failed_types:
        print("\n" + "=" * 80)
        print("NEEDS IMPROVEMENT (Correction not fully working)")
        print("=" * 80)
        for error_type in sorted(failed_types):
            counts = stats["by_error_type"][error_type]
            failed = counts["total"] - counts["correction_success"]
            print(f"  {error_type}: {failed} failures out of {counts['total']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Address Validation Test Runner with Enforcer Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_address_validation.py                           # Default JSON file
  python test_address_validation.py batch_addresses.json      # Custom JSON
  python test_address_validation.py addresses.csv             # CSV file
  python test_address_validation.py data.json --format=json   # JSON output
  python test_address_validation.py data.json --format=csv    # CSV output
  python test_address_validation.py data.json --basic         # Basic mode (no enforcers)
        """,
    )

    parser.add_argument(
        "file",
        nargs="?",
        default="test_addresses_validation.json",
        help="Path to address file (JSON or CSV). Default: test_addresses_validation.json",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["presentation", "json", "csv"],
        default="presentation",
        help="Output format. Default: presentation",
    )

    parser.add_argument(
        "--basic",
        "-b",
        action="store_true",
        help="Run basic validation without enforcer agents",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed per-address output",
    )

    args = parser.parse_args()

    if args.basic:
        run_basic_validation(args.file, verbose=args.verbose or True)
    else:
        asyncio.run(
            run_with_enforcers(
                args.file,
                output_format=args.format,
                verbose=args.verbose,
            )
        )


if __name__ == "__main__":
    main()
