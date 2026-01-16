"""
Databricks SQL Connector - Direct Delta Table Access

Simple connector to pull customer data directly from Databricks Delta tables.
Provides the same interface as CustomerDB for easy swapping.
"""

import os
import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class DatabricksConfig:
    """Configuration for Databricks connection."""
    server_hostname: str  # e.g., "adb-1234567890.1.azuredatabricks.net"
    http_path: str  # e.g., "/sql/1.0/warehouses/abc123"
    access_token: str
    catalog: str = "main"
    schema: str = "default"
    customers_table: str = "customers"


class DatabricksDB:
    """
    Databricks Delta table connector with CustomerDB-compatible interface.

    Usage:
        config = DatabricksConfig(
            server_hostname=os.environ["DATABRICKS_HOST"],
            http_path=os.environ["DATABRICKS_HTTP_PATH"],
            access_token=os.environ["DATABRICKS_TOKEN"],
            catalog="my_catalog",
            schema="icda",
            customers_table="customers"
        )
        db = DatabricksDB(config)

        # Same interface as CustomerDB
        result = db.search(state="TX", limit=10)
        result = db.lookup("CRID-001234")
    """

    STATE_CODE_TO_NAME = {
        "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
        "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
        "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
        "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
        "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
        "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
        "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
        "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
        "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
        "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
        "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
        "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
        "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    }

    STATE_NAME_TO_CODE = {v.lower(): k for k, v in STATE_CODE_TO_NAME.items()}

    def __init__(self, config: DatabricksConfig):
        self.config = config
        self._connection = None
        self._available_states: set[str] = set()
        self.full_table_name = f"{config.catalog}.{config.schema}.{config.customers_table}"

    def _get_connection(self):
        """Lazy connection initialization."""
        if self._connection is None:
            try:
                from databricks import sql
                self._connection = sql.connect(
                    server_hostname=self.config.server_hostname,
                    http_path=self.config.http_path,
                    access_token=self.config.access_token
                )
                print(f"Connected to Databricks: {self.config.server_hostname}")
                self._load_available_states()
            except ImportError:
                raise ImportError(
                    "databricks-sql-connector not installed. "
                    "Run: pip install databricks-sql-connector"
                )
        return self._connection

    def _load_available_states(self):
        """Load list of states with data."""
        cursor = self._connection.cursor()
        try:
            cursor.execute(f"SELECT DISTINCT state FROM {self.full_table_name}")
            self._available_states = {row[0] for row in cursor.fetchall() if row[0]}
            print(f"Available states in Databricks: {sorted(self._available_states)}")
        finally:
            cursor.close()

    def _execute_query(self, sql: str, params: dict = None) -> list[dict]:
        """Execute SQL and return results as list of dicts."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        finally:
            cursor.close()

    def _execute_scalar(self, sql: str, params: dict = None):
        """Execute SQL and return single value."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            cursor.close()

    # =========================================================================
    # CustomerDB-Compatible Interface
    # =========================================================================

    def has_state(self, state_code: str) -> bool:
        """Check if a state exists in the dataset."""
        return state_code.upper() in self._available_states

    def get_available_states(self) -> list[str]:
        """Get list of states with data."""
        return sorted(self._available_states)

    def lookup(self, crid: str) -> dict:
        """Lookup customer by CRID."""
        crid = crid.upper()
        if crid.startswith("CRID-"):
            num = crid.removeprefix("CRID-")
            # Try different formats
            crids_to_try = [
                f"CRID-{num.zfill(6)}",
                f"CRID-{num.zfill(3)}",
                crid
            ]
        else:
            crids_to_try = [crid]

        for crid_fmt in crids_to_try:
            sql = f"""
                SELECT * FROM {self.full_table_name}
                WHERE UPPER(crid) = UPPER(%(crid)s)
                LIMIT 1
            """
            results = self._execute_query(sql, {"crid": crid_fmt})
            if results:
                return {"success": True, "data": results[0]}

        return {"success": False, "error": f"CRID {crid} not found"}

    def search(
        self,
        state: str = None,
        city: str = None,
        min_moves: int = None,
        customer_type: str = None,
        has_apartment: bool = None,
        status: str = None,
        limit: int = None
    ) -> dict:
        """Search customers with filters."""

        # Check if requested state exists
        if state:
            state_upper = state.upper()
            if state_upper not in self._available_states:
                state_name = self.STATE_CODE_TO_NAME.get(state_upper, state_upper)
                return {
                    "success": False,
                    "error": "state_not_available",
                    "message": f"No customer data available for {state_name} ({state_upper}).",
                    "requested_state": state_upper,
                    "available_states": self.get_available_states(),
                }

        # Build query
        conditions = []
        params = {}

        if state:
            conditions.append("UPPER(state) = UPPER(%(state)s)")
            params["state"] = state

        if city:
            conditions.append("LOWER(city) LIKE LOWER(%(city)s)")
            params["city"] = f"%{city}%"

        if min_moves is not None:
            conditions.append("move_count >= %(min_moves)s")
            params["min_moves"] = min_moves

        if customer_type:
            conditions.append("UPPER(customer_type) = UPPER(%(customer_type)s)")
            params["customer_type"] = customer_type

        if has_apartment:
            conditions.append("(LOWER(address) LIKE '%%apt%%' OR LOWER(address) LIKE '%%unit%%')")

        if status:
            conditions.append("UPPER(status) = UPPER(%(status)s)")
            params["status"] = status

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        limit_clause = f"LIMIT {limit}" if limit else ""

        # Get count first
        count_sql = f"SELECT COUNT(*) FROM {self.full_table_name} WHERE {where_clause}"
        total = self._execute_scalar(count_sql, params)

        # Get data
        sql = f"SELECT * FROM {self.full_table_name} WHERE {where_clause} {limit_clause}"
        data = self._execute_query(sql, params)

        return {"success": True, "total": total, "data": data}

    def stats(self) -> dict:
        """Get customer counts by state."""
        sql = f"""
            SELECT state, COUNT(*) as count
            FROM {self.full_table_name}
            GROUP BY state
            ORDER BY count DESC
        """
        results = self._execute_query(sql)

        state_counts = {r["state"]: r["count"] for r in results}
        total = sum(state_counts.values())

        return {"success": True, "data": state_counts, "total": total}

    def count_by_criteria(
        self,
        state: str = None,
        city: str = None,
        status: str = None,
        customer_type: str = None,
        min_moves: int = None
    ) -> dict:
        """Fast count without returning data."""
        conditions = []
        params = {}

        if state:
            state_upper = state.upper()
            if state_upper not in self._available_states:
                return {"success": True, "count": 0, "reason": f"No data for state {state_upper}"}
            conditions.append("UPPER(state) = UPPER(%(state)s)")
            params["state"] = state

        if city:
            conditions.append("LOWER(city) LIKE LOWER(%(city)s)")
            params["city"] = f"%{city}%"

        if status:
            conditions.append("UPPER(status) = UPPER(%(status)s)")
            params["status"] = status

        if customer_type:
            conditions.append("UPPER(customer_type) = UPPER(%(customer_type)s)")
            params["customer_type"] = customer_type

        if min_moves is not None:
            conditions.append("move_count >= %(min_moves)s")
            params["min_moves"] = min_moves

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT COUNT(*) FROM {self.full_table_name} WHERE {where_clause}"

        count = self._execute_scalar(sql, params)

        return {
            "success": True,
            "count": count,
            "filters_applied": {
                "state": state, "city": city, "status": status,
                "customer_type": customer_type, "min_moves": min_moves
            }
        }

    def group_by(
        self,
        field: str,
        state: str = None,
        status: str = None,
        customer_type: str = None
    ) -> dict:
        """Group and count by field."""
        valid_fields = {"state", "status", "customer_type", "city", "move_count"}
        if field not in valid_fields:
            return {"success": False, "error": f"Invalid field. Use one of: {valid_fields}"}

        conditions = []
        params = {}

        if state:
            conditions.append("UPPER(state) = UPPER(%(state)s)")
            params["state"] = state
        if status:
            conditions.append("UPPER(status) = UPPER(%(status)s)")
            params["status"] = status
        if customer_type:
            conditions.append("UPPER(customer_type) = UPPER(%(customer_type)s)")
            params["customer_type"] = customer_type

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
            SELECT {field}, COUNT(*) as count
            FROM {self.full_table_name}
            WHERE {where_clause}
            GROUP BY {field}
            ORDER BY count DESC
        """
        results = self._execute_query(sql, params)

        return {
            "success": True,
            "field": field,
            "total_records": sum(r["count"] for r in results),
            "unique_values": len(results),
            "groups": [{"value": r[field], "count": r["count"]} for r in results]
        }

    def autocomplete(self, field: str, prefix: str, limit: int = 10) -> dict:
        """Prefix-based autocomplete."""
        if field not in ("address", "name", "city"):
            return {"success": False, "error": f"Unknown field: {field}"}

        prefix_lower = prefix.lower().strip()
        if not prefix_lower:
            return {"success": False, "error": "Empty prefix"}

        sql = f"""
            SELECT DISTINCT {field}, crid, name, city, state
            FROM {self.full_table_name}
            WHERE LOWER({field}) LIKE %(prefix)s
            LIMIT %(limit)s
        """
        results = self._execute_query(sql, {"prefix": f"{prefix_lower}%", "limit": limit})

        data = [
            {
                "crid": r["crid"],
                "value": r[field],
                "name": r["name"],
                "city": r["city"],
                "state": r["state"]
            }
            for r in results
        ]

        return {"success": True, "field": field, "prefix": prefix, "count": len(data), "data": data}

    def execute(self, tool: str, query: str) -> dict:
        """Execute a tool with natural language query (basic parsing)."""
        q = query.casefold()

        match tool:
            case "lookup_crid":
                if m := re.search(r"crid[-:\s]*(\d+)", q):
                    return self.lookup(f"CRID-{m.group(1)}")
                return {"success": False, "error": "No CRID found in query"}

            case "search_customers":
                # Parse state
                state = None
                for code in self._available_states:
                    if code.lower() in q:
                        state = code
                        break
                # Parse min_moves
                min_moves = None
                if m := re.search(r"(\d+)\+?\s*(?:times|moves)", q):
                    min_moves = int(m.group(1))
                return self.search(state=state, min_moves=min_moves)

            case "get_stats":
                return self.stats()

        return {"success": False, "error": f"Unknown tool: {tool}"}

    def raw_sql(self, sql: str, params: dict = None) -> dict:
        """Execute raw SQL query (for advanced use cases).

        WARNING: Use parameterized queries to prevent SQL injection.
        """
        try:
            results = self._execute_query(sql, params)
            return {"success": True, "total": len(results), "data": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def close(self):
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            print("Databricks connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_databricks_config_from_cfg() -> DatabricksConfig:
    """Create DatabricksConfig from icda config object."""
    from icda.config import cfg

    if not cfg.is_databricks_configured():
        raise ValueError(
            "Databricks not configured. Set USE_DATABRICKS=true and provide:\n"
            "  DATABRICKS_SERVER_HOSTNAME\n"
            "  DATABRICKS_HTTP_PATH\n"
            "  DATABRICKS_TOKEN"
        )

    return DatabricksConfig(
        server_hostname=cfg.databricks_server_hostname,
        http_path=cfg.databricks_http_path,
        access_token=cfg.databricks_token,
        catalog=cfg.databricks_catalog,
        schema=cfg.databricks_schema,
        customers_table=cfg.databricks_customers_table
    )


def create_customer_db(data_file_path: str = None):
    """Factory function to create the appropriate database based on config.

    If USE_DATABRICKS=true and Databricks is configured, returns DatabricksDB.
    Otherwise, returns the local CustomerDB using JSON file.

    Args:
        data_file_path: Path to JSON file for CustomerDB (default: customer_data.json)

    Returns:
        DatabricksDB or CustomerDB instance
    """
    from icda.config import cfg

    if cfg.is_databricks_configured():
        print("Using Databricks Delta tables for customer data")
        config = get_databricks_config_from_cfg()
        return DatabricksDB(config)
    else:
        print("Using local JSON file for customer data")
        from pathlib import Path
        from icda.database import CustomerDB
        if data_file_path is None:
            data_file_path = "customer_data.json"
        return CustomerDB(Path(data_file_path))


def get_databricks_config_from_env() -> DatabricksConfig:
    """Create DatabricksConfig from environment variables.

    Required env vars:
        DATABRICKS_SERVER_HOSTNAME - e.g., "adb-123.1.azuredatabricks.net"
        DATABRICKS_HTTP_PATH - e.g., "/sql/1.0/warehouses/abc123"
        DATABRICKS_TOKEN - Personal access token or OAuth token

    Optional env vars:
        DATABRICKS_CATALOG - Default: "main"
        DATABRICKS_SCHEMA - Default: "default"
        DATABRICKS_CUSTOMERS_TABLE - Default: "customers"
    """
    hostname = os.environ.get("DATABRICKS_SERVER_HOSTNAME")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH")
    token = os.environ.get("DATABRICKS_TOKEN")

    if not all([hostname, http_path, token]):
        raise ValueError(
            "Missing required environment variables. Set:\n"
            "  DATABRICKS_SERVER_HOSTNAME\n"
            "  DATABRICKS_HTTP_PATH\n"
            "  DATABRICKS_TOKEN"
        )

    return DatabricksConfig(
        server_hostname=hostname,
        http_path=http_path,
        access_token=token,
        catalog=os.environ.get("DATABRICKS_CATALOG", "main"),
        schema=os.environ.get("DATABRICKS_SCHEMA", "default"),
        customers_table=os.environ.get("DATABRICKS_CUSTOMERS_TABLE", "customers")
    )
