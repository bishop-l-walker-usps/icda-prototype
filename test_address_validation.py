"""Standalone address validation test - NO AWS required.

Author: Bishop Walker
"""

import sys
sys.path.insert(0, '.')

from icda.address_normalizer import AddressNormalizer
from icda.address_validator_engine import AddressValidatorEngine, ValidationMode


def test_addresses():
    engine = AddressValidatorEngine()

    test_cases = [
        # PR with urbanization
        "URB Villa Carolina, 123 Calle A, Carolina, PR 00983",
        # PR without URB (should flag warning)
        "123 Calle Sol, San Juan, PR 00901",
        # Mainland US
        "123 Main Street, Arlington, VA 22201",
        # With typos
        "456 Oak Stret, Chicago, Illinios 60601",
        # Partial address
        "789 Elm, 90210",
    ]

    print("=" * 70)
    print("ADDRESS VALIDATION TEST (No AWS Required)")
    print("=" * 70)

    for addr in test_cases:
        print(f"\nInput: {addr}")
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
