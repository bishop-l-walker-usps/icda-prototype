"""
Address corruption utilities for E2E testing.

Generates test addresses by taking real customer addresses and applying
various corruption strategies to test the address verification pipeline.
"""

import json
import random
import string
from pathlib import Path
from typing import Callable


# Common typo patterns (adjacent keys on QWERTY keyboard)
ADJACENT_KEYS = {
    'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'erfcxs', 'e': 'wsdr',
    'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg', 'i': 'uojk', 'j': 'uiknmh',
    'k': 'iojlm', 'l': 'opk', 'm': 'njk', 'n': 'bhjm', 'o': 'iplk',
    'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'wedxza', 't': 'rfgy',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu', 'z': 'asx',
}


def introduce_typo(text: str, count: int = 1) -> str:
    """Introduce realistic typos into text."""
    if not text or len(text) < 3:
        return text

    result = list(text)
    for _ in range(count):
        strategies = [
            _swap_adjacent_chars,
            _replace_with_adjacent_key,
            _double_letter,
            _drop_letter,
        ]
        strategy = random.choice(strategies)
        result = list(strategy(''.join(result)))

    return ''.join(result)


def _swap_adjacent_chars(text: str) -> str:
    """Swap two adjacent characters."""
    if len(text) < 2:
        return text
    pos = random.randint(0, len(text) - 2)
    chars = list(text)
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return ''.join(chars)


def _replace_with_adjacent_key(text: str) -> str:
    """Replace a character with an adjacent key on keyboard."""
    chars = list(text.lower())
    replaceable = [i for i, c in enumerate(chars) if c in ADJACENT_KEYS]
    if not replaceable:
        return text
    pos = random.choice(replaceable)
    adjacent = ADJACENT_KEYS[chars[pos]]
    chars[pos] = random.choice(adjacent)
    return ''.join(chars)


def _double_letter(text: str) -> str:
    """Double a random letter."""
    if not text:
        return text
    pos = random.randint(0, len(text) - 1)
    return text[:pos] + text[pos] + text[pos:]


def _drop_letter(text: str) -> str:
    """Drop a random letter."""
    if len(text) < 3:
        return text
    pos = random.randint(1, len(text) - 2)  # Don't drop first/last
    return text[:pos] + text[pos + 1:]


# ============================================================================
# Corruption Strategies
# ============================================================================


def typo_in_street(address: dict) -> str:
    """Introduce typos in street name."""
    street = address.get("address", "")
    parts = street.split()
    if len(parts) > 1:
        # Typo in street name (second word usually)
        word_idx = min(1, len(parts) - 1)
        if len(parts[word_idx]) > 2:
            parts[word_idx] = introduce_typo(parts[word_idx])

    corrupted_street = ' '.join(parts)
    return f"{corrupted_street}, {address.get('city', '')}, {address.get('state', '')} {address.get('zip', '')}"


def remove_city(address: dict) -> str:
    """Remove city from address."""
    return f"{address.get('address', '')}, {address.get('state', '')} {address.get('zip', '')}"


def remove_state(address: dict) -> str:
    """Remove state from address."""
    return f"{address.get('address', '')}, {address.get('city', '')} {address.get('zip', '')}"


def remove_zip(address: dict) -> str:
    """Remove ZIP from address."""
    return f"{address.get('address', '')}, {address.get('city', '')}, {address.get('state', '')}"


def swap_components(address: dict) -> str:
    """Swap address components around (ZIP as street number, etc.)."""
    zip_code = address.get('zip', '')
    street = address.get('address', '')
    # Put ZIP at beginning, looks like street number
    if street and zip_code:
        parts = street.split()
        if parts and parts[0].isdigit():
            # Replace street number with ZIP
            parts[0] = zip_code[:3]  # Partial ZIP as number
    return f"{' '.join(parts)}, {address.get('city', '')}, {address.get('state', '')}"


def abbreviate_heavily(address: dict) -> str:
    """Remove most components, leave only street number and partial name."""
    street = address.get('address', '')
    parts = street.split()
    if len(parts) >= 2:
        # Just "123 Main" style
        return f"{parts[0]} {parts[1]}"
    return street


def all_caps(address: dict) -> str:
    """Convert entire address to ALL CAPS."""
    full = f"{address.get('address', '')}, {address.get('city', '')}, {address.get('state', '')} {address.get('zip', '')}"
    return full.upper()


def extra_whitespace(address: dict) -> str:
    """Add excessive whitespace between components."""
    parts = [
        address.get('address', ''),
        address.get('city', ''),
        address.get('state', ''),
        address.get('zip', '')
    ]
    # Add random extra spaces
    result = []
    for part in parts:
        words = part.split()
        spaced_words = '   '.join(words)
        result.append(spaced_words)
    return '  ,   '.join(result)


def partial_zip(address: dict) -> str:
    """Truncate ZIP code."""
    zip_code = address.get('zip', '')
    if len(zip_code) >= 4:
        zip_code = zip_code[:-1]  # Remove last digit
    return f"{address.get('address', '')}, {address.get('city', '')}, {address.get('state', '')} {zip_code}"


def state_full_name(address: dict) -> str:
    """Use full state name instead of code."""
    STATE_NAMES = {
        "NV": "Nevada", "CA": "California", "TX": "Texas", "AZ": "Arizona",
        "FL": "Florida", "NY": "New York", "WA": "Washington", "CO": "Colorado",
        "VA": "Virginia", "GA": "Georgia", "NC": "North Carolina", "IL": "Illinois",
        "PA": "Pennsylvania", "OH": "Ohio", "MI": "Michigan",
    }
    state = address.get('state', '')
    full_name = STATE_NAMES.get(state, state)
    return f"{address.get('address', '')}, {address.get('city', '')}, {full_name} {address.get('zip', '')}"


def phonetic_misspelling(address: dict) -> str:
    """Misspell based on phonetics (common mistakes)."""
    street = address.get('address', '')
    city = address.get('city', '')

    # Common phonetic substitutions
    replacements = [
        ('ough', 'off'), ('tion', 'shun'), ('ph', 'f'), ('ck', 'k'),
        ('ee', 'ea'), ('ie', 'y'), ('ou', 'ow'),
    ]

    for old, new in replacements:
        if old in street.lower():
            street = street.lower().replace(old, new, 1)
            break

    return f"{street}, {city}, {address.get('state', '')} {address.get('zip', '')}"


# ============================================================================
# Main Generator
# ============================================================================


# Corruption strategies with weights (higher = more likely)
CORRUPTION_STRATEGIES: list[tuple[Callable, int]] = [
    (typo_in_street, 20),
    (remove_city, 15),
    (remove_state, 10),
    (remove_zip, 10),
    (swap_components, 5),
    (abbreviate_heavily, 15),
    (all_caps, 10),
    (extra_whitespace, 10),
    (partial_zip, 5),
    (state_full_name, 5),
    (phonetic_misspelling, 5),
]


def _weighted_choice(strategies: list[tuple[Callable, int]]) -> Callable:
    """Select a strategy based on weights."""
    total = sum(w for _, w in strategies)
    r = random.randint(1, total)
    cumulative = 0
    for strategy, weight in strategies:
        cumulative += weight
        if r <= cumulative:
            return strategy
    return strategies[0][0]


def generate_corrupted_addresses(customers: list[dict], count: int = 200) -> list[dict]:
    """
    Generate corrupted addresses from real customer data.

    Args:
        customers: List of customer dicts with address, city, state, zip
        count: Number of corrupted addresses to generate

    Returns:
        List of dicts with id, original_crid, corrupted_address, expected_address
    """
    if not customers:
        raise ValueError("No customers provided")

    results = []

    for i in range(count):
        customer = random.choice(customers)

        # Build expected full address
        expected = f"{customer.get('address', '')}, {customer.get('city', '')}, {customer.get('state', '')} {customer.get('zip', '')}"

        # Select corruption strategy
        corruption = _weighted_choice(CORRUPTION_STRATEGIES)

        # Apply corruption
        address_data = {
            "address": customer.get("address", ""),
            "city": customer.get("city", ""),
            "state": customer.get("state", ""),
            "zip": customer.get("zip", ""),
        }
        corrupted = corruption(address_data)

        results.append({
            "id": f"TEST-{i + 1:03d}",
            "original_crid": customer.get("crid", ""),
            "corrupted_address": corrupted.strip(),
            "expected_address": expected.strip(),
            "corruption_type": corruption.__name__,
        })

    return results


def save_to_json(addresses: list[dict], path: Path) -> None:
    """Save generated addresses to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(addresses, f, indent=2)
    print(f"Saved {len(addresses)} addresses to {path}")


def load_customers(path: Path) -> list[dict]:
    """Load customers from JSON file."""
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    # Generate addresses when run directly
    import sys

    project_root = Path(__file__).parent.parent.parent
    customer_path = project_root / "customer_data.json"
    output_path = project_root / "tests" / "fixtures" / "corrupted_addresses.json"

    if not customer_path.exists():
        print(f"Error: {customer_path} not found")
        sys.exit(1)

    customers = load_customers(customer_path)
    print(f"Loaded {len(customers)} customers")

    addresses = generate_corrupted_addresses(customers, count=200)
    save_to_json(addresses, output_path)

    # Print sample
    print("\nSample corrupted addresses:")
    for addr in addresses[:5]:
        print(f"  [{addr['corruption_type']}] {addr['corrupted_address']}")
