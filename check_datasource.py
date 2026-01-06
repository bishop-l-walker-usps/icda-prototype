#!/usr/bin/env python3
"""Check available data sources for ICDA.

Run this script to see what data sources are available and which one
will be used for Titan embeddings.

Usage:
    python check_datasource.py
    python check_datasource.py --c-library-export data/export.json
    python check_datasource.py --c-library-api http://localhost:8080/api/export
"""

import argparse
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from icda.datasource import DataSourceManager, check_data_sources


def main():
    parser = argparse.ArgumentParser(
        description="Check available data sources for ICDA embeddings"
    )
    parser.add_argument(
        "--c-library-export",
        help="Path to C library export file",
        default=os.getenv("C_LIBRARY_EXPORT_PATH"),
    )
    parser.add_argument(
        "--c-library-api",
        help="C library REST API URL",
        default=os.getenv("C_LIBRARY_API_URL"),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable",
    )

    args = parser.parse_args()

    result = check_data_sources(
        project_root=".",
        c_library_export=args.c_library_export,
        c_library_api=args.c_library_api,
    )

    if args.json:
        import json
        # Remove summary for JSON output
        del result["summary"]
        print(json.dumps(result, indent=2))
    else:
        print(result["summary"])

        # Additional guidance
        print("\nUSAGE NOTES:")
        print("-" * 60)

        if result["active_source"] == "customer_json":
            print("Using customer_data.json for TESTING.")
            print("To use C library data instead:")
            print("  1. Delete or rename customer_data.json")
            print("  2. Provide C library export file or API URL")

        elif result["active_source"] == "c_library_file":
            print("Using C library export file for PRODUCTION.")
            print("This data will be embedded with Titan for address validation.")

        elif result["active_source"] == "c_library_api":
            print("Using C library REST API for PRODUCTION.")
            print("Data will be fetched and embedded with Titan.")

        else:
            print("WARNING: No data source available!")
            print("Provide one of:")
            print("  - customer_data.json (for testing)")
            print("  - C library export file (--c-library-export)")
            print("  - C library API URL (--c-library-api)")


if __name__ == "__main__":
    main()
