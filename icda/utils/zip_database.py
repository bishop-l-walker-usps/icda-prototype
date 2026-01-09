"""Dynamic ZIP code database built from customer data.

This module builds and maintains a ZIP→City/State mapping from
the actual customer data, providing much better coverage than
a static dictionary.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ZipInfo:
    """Information about a ZIP code."""
    zip_code: str
    primary_city: str
    state: str
    alternate_cities: list[str]
    occurrence_count: int
    is_puerto_rico: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "zip_code": self.zip_code,
            "primary_city": self.primary_city,
            "state": self.state,
            "alternate_cities": self.alternate_cities,
            "occurrence_count": self.occurrence_count,
            "is_puerto_rico": self.is_puerto_rico,
        }


class ZipDatabase:
    """Dynamic ZIP code database built from customer data.

    Usage:
        # Build from customer data
        zip_db = ZipDatabase()
        zip_db.build_from_customers(customers)

        # Lookup
        info = zip_db.lookup("22201")
        if info:
            print(f"{info.zip_code} -> {info.primary_city}, {info.state}")

        # Save/load for persistence
        zip_db.save_to_file("zip_database.json")
        zip_db.load_from_file("zip_database.json")
    """

    def __init__(self):
        self._data: dict[str, ZipInfo] = {}
        self._state_zips: dict[str, set[str]] = defaultdict(set)
        self._city_zips: dict[str, set[str]] = defaultdict(set)

    def build_from_customers(
        self,
        customers: list[dict[str, Any]],
        zip_field: str = "zip",
        city_field: str = "city",
        state_field: str = "state",
    ) -> dict[str, int]:
        """Build ZIP database from customer records.

        Args:
            customers: List of customer dicts
            zip_field: Field name for ZIP code
            city_field: Field name for city
            state_field: Field name for state

        Returns:
            Statistics dict with counts
        """
        # Collect all ZIP code occurrences
        zip_cities: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        zip_states: dict[str, str] = {}

        for customer in customers:
            zip_code = str(customer.get(zip_field, "")).strip()[:5]
            city = str(customer.get(city_field, "")).strip().title()
            state = str(customer.get(state_field, "")).strip().upper()[:2]

            if not zip_code or len(zip_code) != 5 or not zip_code.isdigit():
                continue

            if city:
                zip_cities[zip_code][city] += 1
            if state and len(state) == 2:
                zip_states[zip_code] = state

        # Build ZipInfo records
        self._data.clear()
        self._state_zips.clear()
        self._city_zips.clear()

        for zip_code, cities in zip_cities.items():
            if not cities:
                continue

            # Find primary city (most occurrences)
            sorted_cities = sorted(cities.items(), key=lambda x: x[1], reverse=True)
            primary_city = sorted_cities[0][0]
            alternate_cities = [c for c, _ in sorted_cities[1:5]]
            total_count = sum(cities.values())

            state = zip_states.get(zip_code, "")
            is_pr = zip_code[:3] in ("006", "007", "008", "009")

            if is_pr and not state:
                state = "PR"

            info = ZipInfo(
                zip_code=zip_code,
                primary_city=primary_city,
                state=state,
                alternate_cities=alternate_cities,
                occurrence_count=total_count,
                is_puerto_rico=is_pr,
            )
            self._data[zip_code] = info

            # Index by state and city
            if state:
                self._state_zips[state].add(zip_code)
            self._city_zips[primary_city.lower()].add(zip_code)

        logger.info(
            f"ZipDatabase built: {len(self._data)} ZIPs, "
            f"{len(self._state_zips)} states, "
            f"{sum(1 for z in self._data.values() if z.is_puerto_rico)} PR ZIPs"
        )

        return {
            "total_zips": len(self._data),
            "total_states": len(self._state_zips),
            "pr_zips": sum(1 for z in self._data.values() if z.is_puerto_rico),
            "total_records_processed": len(customers),
        }

    def lookup(self, zip_code: str) -> Optional[ZipInfo]:
        """Look up a ZIP code."""
        return self._data.get(zip_code[:5] if zip_code else "")

    def get_city_state(self, zip_code: str) -> tuple[str, str] | None:
        """Get city and state for a ZIP code.

        Returns:
            Tuple of (city, state) or None if not found.
        """
        info = self.lookup(zip_code)
        if info:
            return (info.primary_city, info.state)
        return None

    def get_zips_for_state(self, state: str) -> set[str]:
        """Get all ZIP codes for a state."""
        return self._state_zips.get(state.upper(), set())

    def get_zips_for_city(self, city: str) -> set[str]:
        """Get all ZIP codes for a city."""
        return self._city_zips.get(city.lower(), set())

    def validate_zip_state(self, zip_code: str, state: str) -> bool:
        """Check if ZIP code matches state."""
        info = self.lookup(zip_code)
        if info:
            return info.state == state.upper()
        return True  # If not in database, don't invalidate

    def validate_zip_city(self, zip_code: str, city: str) -> bool:
        """Check if ZIP code matches city."""
        info = self.lookup(zip_code)
        if info:
            city_lower = city.lower()
            return (
                info.primary_city.lower() == city_lower or
                any(c.lower() == city_lower for c in info.alternate_cities)
            )
        return True  # If not in database, don't invalidate

    def save_to_file(self, filepath: str | Path) -> int:
        """Save database to JSON file.

        Returns:
            Number of records saved.
        """
        data = {
            "zips": {k: v.to_dict() for k, v in self._data.items()},
            "version": "1.0",
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

        logger.info(f"ZipDatabase saved to {filepath}: {len(self._data)} records")
        return len(self._data)

    def load_from_file(self, filepath: str | Path) -> int:
        """Load database from JSON file.

        Returns:
            Number of records loaded.
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            self._data.clear()
            self._state_zips.clear()
            self._city_zips.clear()

            for zip_code, info_dict in data.get("zips", {}).items():
                info = ZipInfo(
                    zip_code=info_dict["zip_code"],
                    primary_city=info_dict["primary_city"],
                    state=info_dict["state"],
                    alternate_cities=info_dict.get("alternate_cities", []),
                    occurrence_count=info_dict.get("occurrence_count", 0),
                    is_puerto_rico=info_dict.get("is_puerto_rico", False),
                )
                self._data[zip_code] = info

                if info.state:
                    self._state_zips[info.state].add(zip_code)
                self._city_zips[info.primary_city.lower()].add(zip_code)

            logger.info(f"ZipDatabase loaded from {filepath}: {len(self._data)} records")
            return len(self._data)

        except FileNotFoundError:
            logger.warning(f"ZipDatabase file not found: {filepath}")
            return 0
        except Exception as e:
            logger.error(f"Failed to load ZipDatabase: {e}")
            return 0

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        pr_count = sum(1 for z in self._data.values() if z.is_puerto_rico)
        return {
            "total_zips": len(self._data),
            "total_states": len(self._state_zips),
            "total_cities": len(self._city_zips),
            "pr_zips": pr_count,
            "us_zips": len(self._data) - pr_count,
        }

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, zip_code: str) -> bool:
        return zip_code in self._data


# Global instance for app-wide use
zip_database = ZipDatabase()


def build_zip_database_from_file(
    customer_file: str | Path,
    save_path: str | Path | None = None,
) -> dict[str, int]:
    """Build ZIP database from a customer JSON file.

    Args:
        customer_file: Path to customer_data.json
        save_path: Optional path to save the database

    Returns:
        Statistics dict
    """
    with open(customer_file, "r") as f:
        customers = json.load(f)

    stats = zip_database.build_from_customers(customers)

    if save_path:
        zip_database.save_to_file(save_path)

    return stats
