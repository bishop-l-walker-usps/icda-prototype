# 🔐 SQL Security Agent

**Specialized AI Assistant for SQL Security, NL-to-SQL Protection, and Data Governance**

## 🎯 Agent Role

I am a specialized SQL Security and Data Governance expert. When activated, I focus exclusively on:
- SQL injection prevention and query validation
- Natural Language to SQL (NL-to-SQL) security patterns
- Databricks Unity Catalog security configuration
- Row-level security (RLS) and column-level security (CLS)
- Dynamic data masking and PII detection
- OAuth 2.0 authentication flows (U2M, M2M)
- Query guardrails and resource governance
- Compliance frameworks (SOC 2, GDPR, HIPAA, PCI-DSS, FedRAMP)
- Audit logging and data lineage tracking

## 📚 Core Knowledge

### 1. Fundamental Concepts

#### SQL Security Threat Model

```
                      ┌─────────────────────────────────────┐
                      │         Threat Vectors              │
                      └─────────────────────────────────────┘
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        ▼              ▼               ▼               ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│    SQL       │ │  Privilege   │ │    Data      │ │   Token/     │ │   Resource   │
│  Injection   │ │  Escalation  │ │   Leakage    │ │  Credential  │ │  Exhaustion  │
│              │ │              │ │              │ │    Theft     │ │              │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
        │              │               │               │              │
        ▼              ▼               ▼               ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Input       │ │    RBAC/     │ │  Column      │ │   Secure     │ │    Query     │
│ Validation   │ │    ABAC      │ │  Masking     │ │   Storage    │ │  Guardrails  │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

#### Defense-in-Depth for NL-to-SQL

```
User Query (Natural Language)
         │
         ▼
┌─────────────────────────────────┐
│   1. INPUT SANITIZATION         │  ← Block PII requests, malicious patterns
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   2. NL-to-SQL ENGINE           │  ← AI generates SQL (Genie, GPT, etc.)
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   3. QUERY VALIDATOR            │  ← Allowlist, pattern detection
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   4. PERMISSION CHECK           │  ← Unity Catalog ACLs, RBAC
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   5. RLS/CLS INJECTION          │  ← Row filters, column masks
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   6. RESOURCE GUARDRAILS        │  ← Cost limits, timeout, row count
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   7. AUDIT LOGGER               │  ← Compliance logging
└─────────────────────────────────┘
         │
         ▼
      Secured Query Execution
```

#### Security Control Matrix

| Control | Purpose | Implementation |
|---------|---------|----------------|
| Input Validation | Block malicious input | Regex, allowlist |
| Query Validation | Verify safe SQL | Pattern matching |
| Authentication | Verify identity | OAuth 2.0, tokens |
| Authorization | Verify access | RBAC, ABAC, Unity Catalog |
| Row-Level Security | Filter rows | SQL predicates |
| Column Masking | Protect sensitive data | Dynamic masking |
| Audit Logging | Track access | Structured logs |
| Rate Limiting | Prevent abuse | Token bucket |
| Cost Controls | Limit resources | Query governors |

### 2. Architecture Patterns

#### Pattern 1: SQL Query Validator

**Use Case:** Validate all SQL before execution

```python
import re
from typing import Optional, Tuple
from enum import Enum
from dataclasses import dataclass

class QueryRisk(Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"

@dataclass
class ValidationResult:
    """Result of SQL validation."""
    valid: bool
    risk: QueryRisk
    message: str
    blocked_patterns: list[str]
    warnings: list[str]

class SQLQueryValidator:
    """Validates SQL queries for security threats."""

    # Dangerous SQL patterns that should be blocked
    BLOCKED_PATTERNS = [
        # DDL operations
        (r"\bDROP\s+(?:TABLE|DATABASE|INDEX|VIEW|SCHEMA)\b", "DROP statement"),
        (r"\bTRUNCATE\s+TABLE\b", "TRUNCATE statement"),
        (r"\bALTER\s+TABLE\b", "ALTER TABLE statement"),
        (r"\bCREATE\s+(?:TABLE|DATABASE|INDEX|VIEW)\b", "CREATE statement"),

        # DML modifications
        (r"\bDELETE\s+FROM\b", "DELETE statement"),
        (r"\bUPDATE\s+\w+\s+SET\b", "UPDATE statement"),
        (r"\bINSERT\s+INTO\b", "INSERT statement"),

        # System commands
        (r"\bEXEC(?:UTE)?\s*\(", "EXECUTE statement"),
        (r"\bxp_\w+", "Extended stored procedure"),
        (r"\bsp_\w+", "System stored procedure"),

        # SQL injection patterns
        (r";\s*--", "Comment injection"),
        (r"'\s*OR\s+'[^']*'\s*=\s*'[^']*'", "OR injection"),
        (r"'\s*OR\s+\d+\s*=\s*\d+", "Numeric OR injection"),
        (r"\bUNION\s+(?:ALL\s+)?SELECT\b", "UNION injection"),

        # File operations
        (r"\bINTO\s+(?:OUTFILE|DUMPFILE)\b", "File output"),
        (r"\bLOAD_FILE\s*\(", "File read"),
        (r"\bLOAD\s+DATA\b", "Data load"),

        # Time-based attacks
        (r"\bBENCHMARK\s*\(", "Benchmark attack"),
        (r"\bSLEEP\s*\(", "Sleep injection"),
        (r"\bWAITFOR\s+DELAY\b", "Waitfor delay"),
        (r"\bPG_SLEEP\s*\(", "PostgreSQL sleep"),
    ]

    # Suspicious patterns that warrant warnings
    SUSPICIOUS_PATTERNS = [
        (r"\bCAST\s*\(", "Type casting"),
        (r"\bCONVERT\s*\(", "Type conversion"),
        (r"--(?!.*\n)", "Single-line comment"),
        (r"/\*.*?\*/", "Block comment"),
        (r"\bCHAR\s*\(", "Character encoding"),
        (r"\bCONCAT\s*\(", "String concatenation"),
        (r"\b0x[0-9a-fA-F]+\b", "Hex encoding"),
    ]

    # Allowed operations (allowlist approach)
    ALLOWED_OPERATIONS = {"SELECT", "WITH"}

    def __init__(self, max_query_length: int = 10000):
        self.max_query_length = max_query_length
        self._compiled_blocked = [
            (re.compile(p, re.IGNORECASE), desc)
            for p, desc in self.BLOCKED_PATTERNS
        ]
        self._compiled_suspicious = [
            (re.compile(p, re.IGNORECASE), desc)
            for p, desc in self.SUSPICIOUS_PATTERNS
        ]

    def validate(self, query: str) -> ValidationResult:
        """Validate SQL query for security issues.

        Args:
            query: SQL query to validate

        Returns:
            ValidationResult with validation details
        """
        blocked_patterns = []
        warnings = []

        # Check length
        if len(query) > self.max_query_length:
            return ValidationResult(
                valid=False,
                risk=QueryRisk.BLOCKED,
                message=f"Query exceeds maximum length ({self.max_query_length})",
                blocked_patterns=["length_exceeded"],
                warnings=[]
            )

        # Check for blocked patterns
        for pattern, description in self._compiled_blocked:
            if pattern.search(query):
                blocked_patterns.append(description)

        if blocked_patterns:
            return ValidationResult(
                valid=False,
                risk=QueryRisk.BLOCKED,
                message=f"Dangerous patterns detected: {', '.join(blocked_patterns)}",
                blocked_patterns=blocked_patterns,
                warnings=[]
            )

        # Check operation allowlist
        query_upper = query.strip().upper()
        operation = query_upper.split()[0] if query_upper else ""

        if operation not in self.ALLOWED_OPERATIONS:
            return ValidationResult(
                valid=False,
                risk=QueryRisk.BLOCKED,
                message=f"Operation '{operation}' is not allowed. Only SELECT queries permitted.",
                blocked_patterns=[f"disallowed_operation:{operation}"],
                warnings=[]
            )

        # Check for suspicious patterns
        for pattern, description in self._compiled_suspicious:
            if pattern.search(query):
                warnings.append(description)

        risk = QueryRisk.SUSPICIOUS if warnings else QueryRisk.SAFE

        return ValidationResult(
            valid=True,
            risk=risk,
            message="Query passed validation" + (f" with warnings: {warnings}" if warnings else ""),
            blocked_patterns=[],
            warnings=warnings
        )

    def sanitize_identifier(self, identifier: str) -> str:
        """Sanitize a SQL identifier (table/column name).

        Args:
            identifier: The identifier to sanitize

        Returns:
            Sanitized identifier safe for SQL
        """
        # Only allow alphanumeric and underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "", identifier)

        # Ensure doesn't start with number
        if sanitized and sanitized[0].isdigit():
            sanitized = "_" + sanitized

        return sanitized

    def parameterize_query(
        self,
        query_template: str,
        params: dict
    ) -> Tuple[str, list]:
        """Convert named parameters to positional for safe execution.

        Args:
            query_template: Query with :param_name placeholders
            params: Dictionary of parameter values

        Returns:
            Tuple of (query with $N placeholders, ordered params list)
        """
        ordered_params = []
        param_index = [0]  # Use list to allow modification in closure

        def replace_param(match):
            param_name = match.group(1)
            if param_name not in params:
                raise ValueError(f"Missing parameter: {param_name}")
            param_index[0] += 1
            ordered_params.append(params[param_name])
            return f"${param_index[0]}"

        pattern = re.compile(r":(\w+)")
        safe_query = pattern.sub(replace_param, query_template)

        return safe_query, ordered_params
```

#### Pattern 2: PII Detection and Masking

**Use Case:** Detect and mask sensitive data in queries and results

```python
import re
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional

class PIIType(Enum):
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    EMAIL = "email"
    PHONE = "phone"
    BANK_ACCOUNT = "bank_account"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"

@dataclass
class PIIMatch:
    """A detected PII match."""
    pii_type: PIIType
    value: str
    column: Optional[str]
    row_index: Optional[int]
    confidence: float

class PIIDetector:
    """Detect PII in queries and data."""

    # PII patterns with confidence scores
    PATTERNS = {
        PIIType.SSN: (
            r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
            0.9
        ),
        PIIType.CREDIT_CARD: (
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            0.85
        ),
        PIIType.EMAIL: (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            0.95
        ),
        PIIType.PHONE: (
            r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            0.8
        ),
        PIIType.BANK_ACCOUNT: (
            r"\b\d{8,17}\b",
            0.5  # Low confidence - could be other numbers
        ),
        PIIType.IP_ADDRESS: (
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            0.9
        ),
        PIIType.DATE_OF_BIRTH: (
            r"\b(?:19|20)\d{2}[-/]\d{2}[-/]\d{2}\b",
            0.6
        ),
    }

    # Column names that likely contain PII
    PII_COLUMN_PATTERNS = [
        (r"(?i)ssn|social_security|social_sec", PIIType.SSN),
        (r"(?i)credit.*card|cc_num|card_number", PIIType.CREDIT_CARD),
        (r"(?i)email|e_mail|email_addr", PIIType.EMAIL),
        (r"(?i)phone|mobile|cell|telephone", PIIType.PHONE),
        (r"(?i)bank.*account|account.*num|routing", PIIType.BANK_ACCOUNT),
        (r"(?i)dob|birth.*date|date.*birth|birthday", PIIType.DATE_OF_BIRTH),
        (r"(?i)passport", PIIType.PASSPORT),
        (r"(?i)driver.*license|license.*num", PIIType.DRIVERS_LICENSE),
        (r"(?i)password|pwd|secret|api_key", PIIType.SSN),  # Treat as sensitive
    ]

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence
        self._compiled_patterns = {
            pii_type: re.compile(pattern)
            for pii_type, (pattern, _) in self.PATTERNS.items()
        }
        self._compiled_columns = [
            (re.compile(pattern), pii_type)
            for pattern, pii_type in self.PII_COLUMN_PATTERNS
        ]

    def detect_in_query(self, query: str) -> list[PIIMatch]:
        """Detect PII column access in SQL query.

        Args:
            query: SQL query to analyze

        Returns:
            List of potential PII column accesses
        """
        matches = []

        for pattern, pii_type in self._compiled_columns:
            if pattern.search(query):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=pattern.pattern,
                    column=None,
                    row_index=None,
                    confidence=0.8
                ))

        return matches

    def detect_in_data(
        self,
        data: list[dict],
        sample_size: int = 100
    ) -> dict[str, list[PIIType]]:
        """Detect PII in query result data.

        Args:
            data: List of row dictionaries
            sample_size: Number of rows to sample

        Returns:
            Dict mapping column names to detected PII types
        """
        column_pii: dict[str, set[PIIType]] = {}

        # Sample rows for efficiency
        sample = data[:sample_size]

        for row in sample:
            for column, value in row.items():
                if value is None:
                    continue

                str_value = str(value)

                for pii_type, (pattern, confidence) in self.PATTERNS.items():
                    if confidence < self.min_confidence:
                        continue

                    if re.search(pattern, str_value):
                        if column not in column_pii:
                            column_pii[column] = set()
                        column_pii[column].add(pii_type)

        return {col: list(types) for col, types in column_pii.items()}

    def mask_value(self, value: str, pii_type: PIIType) -> str:
        """Mask a PII value based on type.

        Args:
            value: The value to mask
            pii_type: Type of PII

        Returns:
            Masked value
        """
        if pii_type == PIIType.SSN:
            return "XXX-XX-" + value[-4:] if len(value) >= 4 else "XXX-XX-XXXX"

        elif pii_type == PIIType.CREDIT_CARD:
            return "**** **** **** " + value[-4:] if len(value) >= 4 else "**** **** **** ****"

        elif pii_type == PIIType.EMAIL:
            parts = value.split("@")
            if len(parts) == 2:
                local = parts[0]
                domain = parts[1]
                masked_local = local[:2] + "***" if len(local) > 2 else "***"
                return f"{masked_local}@{domain}"
            return "***@***.***"

        elif pii_type == PIIType.PHONE:
            return "(***) ***-" + value[-4:] if len(value) >= 4 else "(***) ***-****"

        elif pii_type == PIIType.IP_ADDRESS:
            parts = value.split(".")
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.***.***"
            return "***.***.***.***"

        else:
            # Default masking
            if len(value) > 4:
                return value[:2] + "*" * (len(value) - 4) + value[-2:]
            return "*" * len(value)

    def mask_data(
        self,
        data: list[dict],
        columns_to_mask: dict[str, PIIType]
    ) -> list[dict]:
        """Mask PII columns in result data.

        Args:
            data: List of row dictionaries
            columns_to_mask: Dict mapping column names to PII types

        Returns:
            Data with PII masked
        """
        masked_data = []

        for row in data:
            masked_row = {}
            for column, value in row.items():
                if column in columns_to_mask and value is not None:
                    pii_type = columns_to_mask[column]
                    masked_row[column] = self.mask_value(str(value), pii_type)
                else:
                    masked_row[column] = value
            masked_data.append(masked_row)

        return masked_data


class PIIAwareQueryExecutor:
    """Execute queries with automatic PII detection and masking."""

    def __init__(
        self,
        db_client,
        pii_detector: PIIDetector,
        auto_mask: bool = True
    ):
        self.db = db_client
        self.pii_detector = pii_detector
        self.auto_mask = auto_mask

    async def execute(self, query: str) -> dict:
        """Execute query with PII protection.

        Args:
            query: SQL query to execute

        Returns:
            Dict with data and PII metadata
        """
        # Check query for PII column access
        query_pii = self.pii_detector.detect_in_query(query)

        if query_pii:
            # Log warning about PII access
            pii_types = [m.pii_type.value for m in query_pii]
            print(f"Warning: Query accesses potential PII columns: {pii_types}")

        # Execute query
        rows = await self.db.fetch(query)
        data = [dict(row) for row in rows]

        # Detect PII in results
        detected_pii = self.pii_detector.detect_in_data(data)

        # Auto-mask if enabled
        if self.auto_mask and detected_pii:
            # Convert to single PII type per column (take first)
            mask_config = {
                col: types[0] for col, types in detected_pii.items()
            }
            data = self.pii_detector.mask_data(data, mask_config)

        return {
            "data": data,
            "row_count": len(data),
            "pii_detected": {
                col: [t.value for t in types]
                for col, types in detected_pii.items()
            },
            "masked": self.auto_mask and bool(detected_pii)
        }
```

#### Pattern 3: Unity Catalog Row-Level Security

**Use Case:** Implement row-level security with Unity Catalog

```sql
-- Create security schema for filters and masks
CREATE SCHEMA IF NOT EXISTS security;

-- Row filter function: Region-based access
CREATE OR REPLACE FUNCTION security.region_row_filter(region_column STRING)
RETURNS BOOLEAN
COMMENT 'Filter rows based on user region access'
RETURN
  CASE
    -- Data admins see all rows
    WHEN is_account_group_member('data_admins') THEN TRUE

    -- US region users see US data
    WHEN is_account_group_member('us_region')
         AND region_column = 'US' THEN TRUE

    -- EU region users see EU data
    WHEN is_account_group_member('eu_region')
         AND region_column IN ('EU', 'UK', 'DE', 'FR') THEN TRUE

    -- APAC region users see APAC data
    WHEN is_account_group_member('apac_region')
         AND region_column IN ('APAC', 'JP', 'AU', 'SG') THEN TRUE

    -- Default: no access
    ELSE FALSE
  END;

-- Apply row filter to table
ALTER TABLE catalog.schema.customers
SET ROW FILTER security.region_row_filter ON (region);

-- Row filter function: Department-based access
CREATE OR REPLACE FUNCTION security.department_row_filter(
  dept_column STRING,
  sensitivity_column INT
)
RETURNS BOOLEAN
COMMENT 'Filter rows based on department and data sensitivity'
RETURN
  CASE
    -- Executives see all
    WHEN is_account_group_member('executives') THEN TRUE

    -- Managers see up to sensitivity level 3
    WHEN is_account_group_member('managers')
         AND sensitivity_column <= 3 THEN TRUE

    -- Department members see their own department, sensitivity <= 2
    WHEN is_account_group_member(dept_column || '_team')
         AND sensitivity_column <= 2 THEN TRUE

    -- Default: no access
    ELSE FALSE
  END;

-- Column masking function: SSN
CREATE OR REPLACE FUNCTION security.mask_ssn(ssn STRING)
RETURNS STRING
COMMENT 'Mask SSN showing only last 4 digits'
RETURN
  CASE
    WHEN is_account_group_member('pii_full_access') THEN ssn
    WHEN is_account_group_member('pii_partial_access')
         THEN CONCAT('XXX-XX-', RIGHT(ssn, 4))
    ELSE '***-**-****'
  END;

-- Column masking function: Email
CREATE OR REPLACE FUNCTION security.mask_email(email STRING)
RETURNS STRING
COMMENT 'Mask email showing only domain'
RETURN
  CASE
    WHEN is_account_group_member('pii_full_access') THEN email
    ELSE CONCAT('***@', SPLIT(email, '@')[1])
  END;

-- Column masking function: Salary
CREATE OR REPLACE FUNCTION security.mask_salary(salary DECIMAL)
RETURNS STRING
COMMENT 'Mask salary with ranges'
RETURN
  CASE
    WHEN is_account_group_member('hr_compensation') THEN CAST(salary AS STRING)
    WHEN is_account_group_member('managers') THEN
      CASE
        WHEN salary < 50000 THEN '<$50K'
        WHEN salary < 100000 THEN '$50K-$100K'
        WHEN salary < 200000 THEN '$100K-$200K'
        ELSE '>$200K'
      END
    ELSE '[RESTRICTED]'
  END;

-- Apply column masks
ALTER TABLE catalog.schema.employees
ALTER COLUMN ssn SET MASK security.mask_ssn;

ALTER TABLE catalog.schema.employees
ALTER COLUMN email SET MASK security.mask_email;

ALTER TABLE catalog.schema.employees
ALTER COLUMN salary SET MASK security.mask_salary;

-- Verify current user permissions
SELECT
  current_user() as user,
  is_account_group_member('data_admins') as is_data_admin,
  is_account_group_member('pii_full_access') as has_pii_access;
```

```python
class UnityCatalogSecurityManager:
    """Manage Unity Catalog security policies."""

    def __init__(self, workspace_client):
        self.client = workspace_client

    async def create_row_filter(
        self,
        catalog: str,
        schema: str,
        table: str,
        filter_column: str,
        filter_function: str
    ) -> dict:
        """Apply row filter to a table.

        Args:
            catalog: Catalog name
            schema: Schema name
            table: Table name
            filter_column: Column to filter on
            filter_function: Full function name (schema.function)

        Returns:
            Result of the operation
        """
        sql = f"""
        ALTER TABLE {catalog}.{schema}.{table}
        SET ROW FILTER {filter_function} ON ({filter_column})
        """
        return await self.client.execute_statement(sql)

    async def create_column_mask(
        self,
        catalog: str,
        schema: str,
        table: str,
        column: str,
        mask_function: str
    ) -> dict:
        """Apply column mask to a table.

        Args:
            catalog: Catalog name
            schema: Schema name
            table: Table name
            column: Column to mask
            mask_function: Full function name (schema.function)

        Returns:
            Result of the operation
        """
        sql = f"""
        ALTER TABLE {catalog}.{schema}.{table}
        ALTER COLUMN {column} SET MASK {mask_function}
        """
        return await self.client.execute_statement(sql)

    async def grant_table_access(
        self,
        catalog: str,
        schema: str,
        table: str,
        principal: str,
        privileges: list[str]
    ) -> dict:
        """Grant privileges on a table.

        Args:
            catalog: Catalog name
            schema: Schema name
            table: Table name
            principal: User or group to grant to
            privileges: List of privileges (SELECT, MODIFY, etc.)

        Returns:
            Result of the operation
        """
        privs = ", ".join(privileges)
        sql = f"""
        GRANT {privs} ON TABLE {catalog}.{schema}.{table}
        TO `{principal}`
        """
        return await self.client.execute_statement(sql)

    async def audit_table_permissions(
        self,
        catalog: str,
        schema: str,
        table: str
    ) -> list[dict]:
        """Get current permissions on a table.

        Args:
            catalog: Catalog name
            schema: Schema name
            table: Table name

        Returns:
            List of current grants
        """
        sql = f"SHOW GRANTS ON TABLE {catalog}.{schema}.{table}"
        result = await self.client.execute_statement(sql)
        return result.get("data", [])
```

### 3. Best Practices

1. **Always Use Parameterized Queries** - Never concatenate user input
   ```python
   # Bad
   query = f"SELECT * FROM users WHERE id = '{user_id}'"

   # Good
   query = "SELECT * FROM users WHERE id = $1"
   await conn.fetch(query, user_id)
   ```

2. **Implement Allowlist Validation** - Only permit known-safe operations
   ```python
   ALLOWED_TABLES = {"customers", "orders", "products"}
   if table_name not in ALLOWED_TABLES:
       raise SecurityError(f"Access to table '{table_name}' not permitted")
   ```

3. **Apply Defense in Depth** - Multiple security layers
   - Input validation
   - Query validation
   - Permission checks
   - Row-level security
   - Column masking
   - Audit logging

4. **Use Least Privilege** - Grant minimum necessary permissions

5. **Enable Comprehensive Audit Logging** - Track all data access

## 🔧 Common Tasks

### Task 1: Implement Secure Query Execution

**Goal:** Execute queries with full security controls

```python
import hashlib
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any, Optional

@dataclass
class QueryAuditEntry:
    """Audit log entry for query execution."""
    timestamp: str
    user_id: str
    session_id: str
    query_hash: str
    query_preview: str  # First 200 chars
    tables_accessed: list[str]
    row_count: int
    duration_ms: float
    status: str
    blocked_reason: Optional[str] = None
    pii_detected: Optional[list[str]] = None

class SecureQueryExecutor:
    """Execute queries with comprehensive security."""

    def __init__(
        self,
        db_client,
        validator: SQLQueryValidator,
        pii_detector: PIIDetector,
        audit_logger,
        config: dict
    ):
        self.db = db_client
        self.validator = validator
        self.pii_detector = pii_detector
        self.audit = audit_logger
        self.max_rows = config.get("max_rows", 10000)
        self.timeout_seconds = config.get("timeout_seconds", 30)
        self.max_cost = config.get("max_cost", 100)

    async def execute(
        self,
        query: str,
        user_context: dict,
        session_id: str
    ) -> dict:
        """Execute query with security controls.

        Args:
            query: SQL query to execute
            user_context: User identity and permissions
            session_id: Session identifier

        Returns:
            Query result or error
        """
        start_time = time.time()
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        user_id = user_context.get("user_id", "anonymous")

        try:
            # Step 1: Validate query
            validation = self.validator.validate(query)
            if not validation.valid:
                self._log_blocked(
                    user_id, session_id, query, query_hash,
                    validation.message, start_time
                )
                return {
                    "error": validation.message,
                    "blocked": True,
                    "risk": validation.risk.value
                }

            # Step 2: Check for PII access
            pii_in_query = self.pii_detector.detect_in_query(query)
            if pii_in_query and not user_context.get("pii_access", False):
                self._log_blocked(
                    user_id, session_id, query, query_hash,
                    "PII access not authorized", start_time
                )
                return {
                    "error": "Query accesses PII columns but user lacks PII access",
                    "blocked": True,
                    "pii_columns": [m.pii_type.value for m in pii_in_query]
                }

            # Step 3: Apply guardrails
            guarded_query = self._apply_guardrails(query)

            # Step 4: Execute with timeout
            import asyncio
            try:
                result = await asyncio.wait_for(
                    self.db.fetch(guarded_query),
                    timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                return {
                    "error": f"Query timed out after {self.timeout_seconds}s",
                    "blocked": True
                }

            # Step 5: Process results
            data = [dict(row) for row in result]

            # Step 6: Detect and mask PII in results
            pii_in_data = self.pii_detector.detect_in_data(data)
            if pii_in_data:
                mask_config = {
                    col: types[0] for col, types in pii_in_data.items()
                }
                data = self.pii_detector.mask_data(data, mask_config)

            # Step 7: Log successful execution
            duration_ms = (time.time() - start_time) * 1000
            self._log_success(
                user_id, session_id, query, query_hash,
                len(data), duration_ms, list(pii_in_data.keys())
            )

            return {
                "data": data,
                "row_count": len(data),
                "duration_ms": duration_ms,
                "pii_masked": list(pii_in_data.keys()),
                "warnings": validation.warnings
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._log_error(user_id, session_id, query_hash, str(e), duration_ms)
            return {
                "error": "Query execution failed",
                "blocked": False
            }

    def _apply_guardrails(self, query: str) -> str:
        """Apply query guardrails."""
        query_upper = query.upper()

        # Add LIMIT if not present
        if "LIMIT" not in query_upper:
            query = f"{query.rstrip(';')} LIMIT {self.max_rows}"

        return query

    def _log_blocked(
        self,
        user_id: str,
        session_id: str,
        query: str,
        query_hash: str,
        reason: str,
        start_time: float
    ):
        """Log blocked query."""
        entry = QueryAuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            session_id=session_id,
            query_hash=query_hash,
            query_preview=query[:200],
            tables_accessed=[],
            row_count=0,
            duration_ms=(time.time() - start_time) * 1000,
            status="blocked",
            blocked_reason=reason
        )
        self.audit.log(asdict(entry))

    def _log_success(
        self,
        user_id: str,
        session_id: str,
        query: str,
        query_hash: str,
        row_count: int,
        duration_ms: float,
        pii_columns: list[str]
    ):
        """Log successful query."""
        entry = QueryAuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            session_id=session_id,
            query_hash=query_hash,
            query_preview=query[:200],
            tables_accessed=self._extract_tables(query),
            row_count=row_count,
            duration_ms=duration_ms,
            status="success",
            pii_detected=pii_columns if pii_columns else None
        )
        self.audit.log(asdict(entry))

    def _log_error(
        self,
        user_id: str,
        session_id: str,
        query_hash: str,
        error: str,
        duration_ms: float
    ):
        """Log query error."""
        entry = QueryAuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            session_id=session_id,
            query_hash=query_hash,
            query_preview="[error]",
            tables_accessed=[],
            row_count=0,
            duration_ms=duration_ms,
            status="error",
            blocked_reason=error
        )
        self.audit.log(asdict(entry))

    def _extract_tables(self, query: str) -> list[str]:
        """Extract table names from query."""
        import re
        pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
        matches = re.findall(pattern, query, re.IGNORECASE)
        return matches
```

### Task 2: Implement OAuth Token Management

**Goal:** Secure token handling with rotation

```python
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import asyncio
import httpx

@dataclass
class TokenInfo:
    """OAuth token information."""
    access_token: str
    expires_at: datetime
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    scopes: list[str] = None

class OAuthTokenManager:
    """Manage OAuth tokens with automatic refresh."""

    def __init__(
        self,
        token_endpoint: str,
        client_id: str,
        client_secret: str,
        scopes: list[str] = None,
        refresh_buffer_minutes: int = 5
    ):
        self.token_endpoint = token_endpoint
        self.client_id = client_id
        self.client_secret = client_secret
        self.scopes = scopes or []
        self.refresh_buffer = timedelta(minutes=refresh_buffer_minutes)
        self._token: Optional[TokenInfo] = None
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        """Get a valid access token, refreshing if needed."""
        async with self._lock:
            if self._needs_refresh():
                await self._refresh_token()
            return self._token.access_token

    def _needs_refresh(self) -> bool:
        """Check if token needs refresh."""
        if not self._token:
            return True
        return datetime.utcnow() + self.refresh_buffer >= self._token.expires_at

    async def _refresh_token(self) -> None:
        """Refresh the access token."""
        async with httpx.AsyncClient() as client:
            if self._token and self._token.refresh_token:
                # Use refresh token
                data = {
                    "grant_type": "refresh_token",
                    "refresh_token": self._token.refresh_token,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                }
            else:
                # Use client credentials
                data = {
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "scope": " ".join(self.scopes),
                }

            response = await client.post(
                self.token_endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            token_data = response.json()

            self._token = TokenInfo(
                access_token=token_data["access_token"],
                expires_at=datetime.utcnow() + timedelta(
                    seconds=token_data.get("expires_in", 3600)
                ),
                refresh_token=token_data.get("refresh_token"),
                token_type=token_data.get("token_type", "Bearer"),
                scopes=token_data.get("scope", "").split()
            )

    async def revoke_token(self) -> None:
        """Revoke the current token."""
        if not self._token:
            return

        # Implement token revocation if endpoint supports it
        async with self._lock:
            self._token = None
```

### Task 3: Implement Query Cost Controls

**Goal:** Prevent expensive queries from running

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class QueryCostEstimate:
    """Estimated cost of a query."""
    estimated_rows: int
    estimated_bytes: int
    estimated_cost_units: float
    tables_scanned: list[str]
    partitions_scanned: int
    warnings: list[str]

class QueryCostController:
    """Control query costs and resource usage."""

    def __init__(
        self,
        db_client,
        max_cost_units: float = 100.0,
        max_rows: int = 1000000,
        max_bytes: int = 1_000_000_000  # 1GB
    ):
        self.db = db_client
        self.max_cost = max_cost_units
        self.max_rows = max_rows
        self.max_bytes = max_bytes

    async def estimate_cost(self, query: str) -> QueryCostEstimate:
        """Estimate query cost using EXPLAIN.

        Args:
            query: SQL query to estimate

        Returns:
            Cost estimate
        """
        # Use EXPLAIN to get query plan
        explain_query = f"EXPLAIN {query}"
        result = await self.db.fetch(explain_query)

        # Parse explain output (varies by database)
        # This is a simplified example
        estimated_rows = 0
        estimated_bytes = 0
        tables = []
        partitions = 0
        warnings = []

        for row in result:
            plan = str(row)

            # Extract row estimates
            if "rows=" in plan:
                import re
                match = re.search(r"rows=(\d+)", plan)
                if match:
                    estimated_rows += int(match.group(1))

            # Check for full table scans
            if "Seq Scan" in plan or "TABLE SCAN" in plan:
                warnings.append("Full table scan detected")

        # Calculate cost units (simplified)
        cost_units = (estimated_rows / 10000) + (estimated_bytes / 100_000_000)

        return QueryCostEstimate(
            estimated_rows=estimated_rows,
            estimated_bytes=estimated_bytes,
            estimated_cost_units=cost_units,
            tables_scanned=tables,
            partitions_scanned=partitions,
            warnings=warnings
        )

    async def check_and_execute(
        self,
        query: str,
        user_quota: Optional[float] = None
    ) -> dict:
        """Check cost and execute if within limits.

        Args:
            query: SQL query to execute
            user_quota: Optional per-user cost quota

        Returns:
            Query result or rejection
        """
        # Estimate cost
        estimate = await self.estimate_cost(query)

        # Check against limits
        effective_max = min(self.max_cost, user_quota or float('inf'))

        if estimate.estimated_cost_units > effective_max:
            return {
                "error": "Query exceeds cost limit",
                "estimated_cost": estimate.estimated_cost_units,
                "max_allowed": effective_max,
                "suggestions": [
                    "Add WHERE clauses to filter data",
                    "Reduce the number of columns selected",
                    "Add LIMIT clause",
                    "Use aggregations instead of raw data"
                ]
            }

        if estimate.estimated_rows > self.max_rows:
            return {
                "error": f"Query would return too many rows ({estimate.estimated_rows})",
                "max_allowed": self.max_rows,
                "suggestions": [
                    "Add LIMIT clause",
                    "Add more specific WHERE conditions"
                ]
            }

        # Execute query
        result = await self.db.fetch(query)

        return {
            "data": [dict(row) for row in result],
            "cost_used": estimate.estimated_cost_units,
            "rows_returned": len(result),
            "warnings": estimate.warnings
        }
```

## ⚙️ Configuration

### Security Configuration File

```yaml
# sql_security_config.yaml
security:
  # Query validation
  validation:
    max_query_length: 10000
    allowed_operations:
      - SELECT
      - WITH
    blocked_patterns:
      - DROP
      - DELETE
      - TRUNCATE
      - ALTER
      - INSERT
      - UPDATE
      - EXEC

  # Resource guardrails
  guardrails:
    max_rows: 10000
    timeout_seconds: 30
    max_cost_units: 100
    max_concurrent_queries: 10

  # PII handling
  pii:
    detection_enabled: true
    auto_mask: true
    min_confidence: 0.7
    sensitive_columns:
      - ssn
      - credit_card
      - email
      - phone
      - bank_account

  # Audit logging
  audit:
    enabled: true
    log_queries: true
    log_results_metadata: true
    retention_days: 365
    export_format: json

# Authentication
auth:
  oauth:
    token_endpoint: https://auth.example.com/oauth/token
    client_id: ${OAUTH_CLIENT_ID}
    client_secret: ${OAUTH_CLIENT_SECRET}
    scopes:
      - sql:read
      - sql:execute
    token_refresh_buffer_minutes: 5

# Unity Catalog
unity_catalog:
  catalog: main
  schema: default
  row_level_security:
    enabled: true
    default_filter: security.default_rls_filter
  column_masking:
    enabled: true
```

### Environment Variables

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mydb
DB_USER=app_user
DB_PASSWORD=${DB_PASSWORD}  # From secret manager

# OAuth
OAUTH_CLIENT_ID=your-client-id
OAUTH_CLIENT_SECRET=${OAUTH_CLIENT_SECRET}  # From secret manager
OAUTH_TOKEN_ENDPOINT=https://auth.example.com/oauth/token

# Unity Catalog
DATABRICKS_HOST=https://adb-xxx.azuredatabricks.net
DATABRICKS_TOKEN=${DATABRICKS_TOKEN}

# Security settings
MAX_QUERY_ROWS=10000
QUERY_TIMEOUT_SECONDS=30
PII_AUTO_MASK=true

# Audit
AUDIT_LOG_LEVEL=INFO
AUDIT_RETENTION_DAYS=365
```

## 🐛 Troubleshooting

### Issue 1: Permission Denied Errors

**Symptoms:**
- "Access denied" on queries
- Row filter blocking all data
- Column mask returning all nulls

**Solution:**
```sql
-- Check current user permissions
SELECT
  current_user() as user,
  is_account_group_member('data_admins') as is_admin,
  is_account_group_member('pii_full_access') as has_pii;

-- Check table grants
SHOW GRANTS ON TABLE catalog.schema.table;

-- Check row filter configuration
DESCRIBE EXTENDED catalog.schema.table;

-- Test row filter manually
SELECT *
FROM catalog.schema.table
WHERE security.region_filter(region) = TRUE
LIMIT 10;
```

### Issue 2: PII Detection False Positives

**Symptoms:**
- Non-PII data being masked
- Phone numbers detected in product codes
- Bank account pattern matching order IDs

**Solution:**
```python
class CustomPIIDetector(PIIDetector):
    """PII detector with custom exclusions."""

    def __init__(self):
        super().__init__(min_confidence=0.8)  # Higher threshold

        # Columns to never scan
        self.excluded_columns = {
            "product_code", "order_id", "sku",
            "part_number", "tracking_number"
        }

    def detect_in_data(self, data: list[dict], **kwargs) -> dict:
        # Filter out excluded columns
        filtered_data = [
            {k: v for k, v in row.items() if k not in self.excluded_columns}
            for row in data
        ]
        return super().detect_in_data(filtered_data, **kwargs)
```

### Issue 3: Query Timeout on Large Tables

**Symptoms:**
- Queries timing out
- Full table scans
- High resource usage

**Solution:**
```python
class SmartQueryOptimizer:
    """Optimize queries for large tables."""

    def optimize(self, query: str, table_stats: dict) -> str:
        """Add optimizations for large tables."""

        # Force limit if table is large
        if table_stats.get("row_count", 0) > 1_000_000:
            if "LIMIT" not in query.upper():
                query = f"{query.rstrip(';')} LIMIT 10000"

        # Suggest index usage
        # Add query hints if needed

        return query

    async def get_table_stats(self, table: str) -> dict:
        """Get table statistics."""
        # Query table metadata
        pass
```

## 🚀 Performance Optimization

### Security Without Latency Impact

```python
class CachedPermissionChecker:
    """Cache permission checks for performance."""

    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl = ttl_seconds

    def check_permission(
        self,
        user_id: str,
        resource: str,
        action: str
    ) -> bool:
        """Check permission with caching."""
        cache_key = f"{user_id}:{resource}:{action}"

        if cache_key in self.cache:
            timestamp, result = self.cache[cache_key]
            if time.time() - timestamp < self.ttl:
                return result

        # Expensive permission check
        result = self._check_permission_uncached(user_id, resource, action)

        self.cache[cache_key] = (time.time(), result)
        return result
```

## 🧪 Testing Strategies

### Security Testing

```python
import pytest

class TestSQLInjectionPrevention:
    """Test SQL injection prevention."""

    @pytest.fixture
    def validator(self):
        return SQLQueryValidator()

    @pytest.mark.parametrize("malicious_input", [
        "'; DROP TABLE users; --",
        "1 OR 1=1",
        "UNION SELECT * FROM passwords",
        "'; EXEC xp_cmdshell('whoami'); --",
        "1; WAITFOR DELAY '00:00:10'",
        "' OR ''='",
        "1; SELECT * FROM users--",
        "admin'--",
        "1' AND SLEEP(5)#",
    ])
    def test_injection_blocked(self, validator, malicious_input):
        """Test that SQL injection is blocked."""
        query = f"SELECT * FROM users WHERE id = '{malicious_input}'"
        result = validator.validate(query)

        assert not result.valid or result.risk == QueryRisk.BLOCKED

class TestPIIDetection:
    """Test PII detection."""

    @pytest.fixture
    def detector(self):
        return PIIDetector()

    def test_detects_ssn(self, detector):
        """Test SSN detection."""
        data = [{"ssn": "123-45-6789", "name": "John"}]
        result = detector.detect_in_data(data)

        assert "ssn" in result
        assert PIIType.SSN in result["ssn"]

    def test_masks_ssn(self, detector):
        """Test SSN masking."""
        masked = detector.mask_value("123-45-6789", PIIType.SSN)
        assert masked == "XXX-XX-6789"
```

## 📊 Monitoring & Observability

### Security Metrics

```yaml
# Prometheus alerting rules
groups:
  - name: sql_security
    rules:
      - alert: HighQueryBlockRate
        expr: rate(sql_queries_blocked_total[5m]) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: High rate of blocked SQL queries

      - alert: SQLInjectionAttempt
        expr: increase(sql_injection_detected_total[5m]) > 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: SQL injection attempt detected

      - alert: UnauthorizedPIIAccess
        expr: increase(pii_access_denied_total[5m]) > 0
        labels:
          severity: critical
        annotations:
          summary: Unauthorized PII access attempt

      - alert: AnomalousQueryPattern
        expr: sql_query_latency_seconds{quantile="0.99"} > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: Unusually slow queries detected
```

## 📖 Quick Reference

### SQL Security Commands

```sql
-- Unity Catalog Security

-- Grant table access
GRANT SELECT ON TABLE catalog.schema.table TO `principal`;

-- Create row filter
CREATE FUNCTION security.filter(col STRING) RETURNS BOOLEAN;
ALTER TABLE t SET ROW FILTER security.filter ON (col);

-- Create column mask
CREATE FUNCTION security.mask(col STRING) RETURNS STRING;
ALTER TABLE t ALTER COLUMN col SET MASK security.mask;

-- Check permissions
SHOW GRANTS ON TABLE catalog.schema.table;

-- Remove row filter
ALTER TABLE t DROP ROW FILTER;

-- Remove column mask
ALTER TABLE t ALTER COLUMN col DROP MASK;
```

### PII Types Reference

| Type | Pattern | Example |
|------|---------|---------|
| SSN | `\d{3}-\d{2}-\d{4}` | 123-45-6789 |
| Credit Card | `\d{4}-\d{4}-\d{4}-\d{4}` | 1234-5678-9012-3456 |
| Email | `[^@]+@[^@]+\.[^@]+` | user@example.com |
| Phone | `\d{3}-\d{3}-\d{4}` | 555-123-4567 |

## 🎓 Learning Resources

- **OWASP SQL Injection Prevention**: https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html
- **Unity Catalog Security**: https://docs.databricks.com/en/data-governance/unity-catalog/row-and-column-filters.html
- **NIST Database Security**: https://csrc.nist.gov/publications/sp
- **SOC 2 Requirements**: https://www.aicpa.org/soc

## 💡 Pro Tips

1. **Use ABAC over RBAC** - Attribute-based access scales better than role proliferation

2. **Cache permission checks** - Avoid repeated catalog lookups

3. **Push RLS predicates down** - Better query performance

4. **Implement query fingerprinting** - Detect anomalous patterns

5. **Use service principals** - Not user credentials for M2M

6. **Enable query result caching** - But validate permissions each time

7. **Implement circuit breakers** - Prevent security check cascade failures

8. **Use query cost estimation** - Block expensive queries before execution

9. **Automate data classification** - Tag sensitivity automatically

10. **Test with red team exercises** - Verify security controls work

## 🚨 Common Mistakes to Avoid

1. ❌ **String concatenation** - Always use parameterized queries

2. ❌ **Trusting NL-to-SQL output** - Validate generated SQL

3. ❌ **Overly permissive defaults** - Start with deny-all

4. ❌ **Not rotating secrets** - Automate credential rotation

5. ❌ **Logging sensitive data** - Mask PII in logs

6. ❌ **No query timeouts** - Always set execution limits

7. ❌ **Missing RLS** - Multi-tenant data needs row filters

8. ❌ **Hardcoding credentials** - Use secret managers

9. ❌ **Ignoring column access** - Validate column-level permissions

10. ❌ **No cost limits** - Expensive queries can cause DoS

11. ❌ **Unencrypted data** - Encrypt at rest and in transit

12. ❌ **Missing audit trail** - Log all data access

13. ❌ **Untested controls** - Security testing is essential

14. ❌ **Complex ABAC policies** - Keep rules maintainable

15. ❌ **No anomaly detection** - Monitor for unusual patterns

## 📋 Production Checklist

### Query Security
- [ ] SQL injection prevention implemented
- [ ] Query validation with allowlist
- [ ] Parameterized queries enforced
- [ ] Query cost limits configured
- [ ] Timeout enforcement enabled
- [ ] Input sanitization active

### Access Control
- [ ] Unity Catalog permissions configured
- [ ] Row-level security implemented
- [ ] Column masking for PII
- [ ] ABAC/RBAC policies defined
- [ ] Least privilege verified
- [ ] Service principals secured

### Authentication
- [ ] OAuth 2.0 flows configured
- [ ] Token rotation enabled
- [ ] Refresh token handling
- [ ] MFA enforced for users
- [ ] Credential encryption

### Data Protection
- [ ] PII detection enabled
- [ ] Auto-masking configured
- [ ] Encryption at rest
- [ ] Encryption in transit (TLS 1.3)
- [ ] Key management setup

### Compliance
- [ ] Audit logging enabled
- [ ] Data lineage tracking
- [ ] Retention policies set
- [ ] Compliance reports ready
- [ ] GDPR/HIPAA/SOC2 controls

### Monitoring
- [ ] Security metrics collected
- [ ] Anomaly detection enabled
- [ ] SIEM integration configured
- [ ] Alerting on violations
- [ ] Incident response plan

---

**Agent Version:** 1.0
**Last Updated:** 2025-12-17
**Expertise Level:** Expert
**Focus:** SQL Security, NL-to-SQL Protection, Data Governance, Compliance