"""
ICDA Address Validator Service
==============================
Validates addresses using the RAG index and knowledge base.
Returns validation results with detailed error information.

Author: Bishop Walker / Salt Water Coder
Project: ICDA Prototype
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationErrorType(str, Enum):
    """Types of validation errors"""
    INVALID_STREET = "invalid_street"
    INVALID_CITY = "invalid_city"
    INVALID_STATE = "invalid_state"
    INVALID_ZIP = "invalid_zip"
    ZIP_CITY_MISMATCH = "zip_city_mismatch"
    ZIP_STATE_MISMATCH = "zip_state_mismatch"
    MISSING_REQUIRED = "missing_required"
    FORMAT_ERROR = "format_error"
    NOT_FOUND = "not_found"


@dataclass
class ValidationError:
    """A single validation error"""
    error_type: ValidationErrorType
    field: str
    message: str
    original_value: Any = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of address validation"""
    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    confidence: float = 0.0
    context: dict = field(default_factory=dict)
    matched_address: Optional[dict] = None
    
    @property
    def error_message(self) -> Optional[str]:
        if self.is_valid:
            return None
        return "; ".join(e.message for e in self.errors)
    
    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": [
                {
                    "type": e.error_type.value,
                    "field": e.field,
                    "message": e.message,
                    "original_value": e.original_value,
                    "suggestion": e.suggestion
                }
                for e in self.errors
            ],
            "confidence": self.confidence,
            "matched_address": self.matched_address
        }


class USStateValidator:
    """Validates US state codes and names"""
    
    STATES = {
        "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
        "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
        "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
        "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
        "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
        "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
        "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
        "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
        "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
        "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
        "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
        "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
        "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
        "PR": "Puerto Rico", "VI": "Virgin Islands", "GU": "Guam"
    }
    
    # Reverse lookup
    STATE_NAMES_TO_CODES = {v.lower(): k for k, v in STATES.items()}
    
    @classmethod
    def is_valid_state(cls, state: str) -> bool:
        """Check if state code or name is valid"""
        state_upper = state.strip().upper()
        state_lower = state.strip().lower()
        
        return state_upper in cls.STATES or state_lower in cls.STATE_NAMES_TO_CODES
    
    @classmethod
    def normalize_state(cls, state: str) -> Optional[str]:
        """Normalize state to 2-letter code"""
        state_upper = state.strip().upper()
        state_lower = state.strip().lower()
        
        if state_upper in cls.STATES:
            return state_upper
        if state_lower in cls.STATE_NAMES_TO_CODES:
            return cls.STATE_NAMES_TO_CODES[state_lower]
        return None
    
    @classmethod
    def get_similar_states(cls, state: str, threshold: float = 0.6) -> list[tuple[str, str, float]]:
        """Find similar state names/codes"""
        from difflib import SequenceMatcher
        
        state_lower = state.strip().lower()
        matches = []
        
        for code, name in cls.STATES.items():
            # Check code similarity
            code_score = SequenceMatcher(None, state_lower, code.lower()).ratio()
            name_score = SequenceMatcher(None, state_lower, name.lower()).ratio()
            
            best_score = max(code_score, name_score)
            if best_score >= threshold:
                matches.append((code, name, best_score))
        
        return sorted(matches, key=lambda x: x[2], reverse=True)


class ZipCodeValidator:
    """Validates and parses ZIP codes"""
    
    # Basic ZIP patterns
    ZIP5_PATTERN = re.compile(r"^\d{5}$")
    ZIP9_PATTERN = re.compile(r"^\d{5}-?\d{4}$")
    
    @classmethod
    def is_valid_format(cls, zip_code: str) -> bool:
        """Check if ZIP code has valid format"""
        zip_clean = zip_code.strip()
        return bool(cls.ZIP5_PATTERN.match(zip_clean) or cls.ZIP9_PATTERN.match(zip_clean))
    
    @classmethod
    def extract_zip5(cls, zip_code: str) -> Optional[str]:
        """Extract 5-digit ZIP from any format"""
        zip_clean = zip_code.strip().replace("-", "")
        if len(zip_clean) >= 5 and zip_clean[:5].isdigit():
            return zip_clean[:5]
        return None
    
    @classmethod
    def normalize(cls, zip_code: str) -> Optional[str]:
        """Normalize ZIP to standard format"""
        zip_clean = zip_code.strip().replace("-", "").replace(" ", "")
        
        if len(zip_clean) == 5 and zip_clean.isdigit():
            return zip_clean
        if len(zip_clean) == 9 and zip_clean.isdigit():
            return f"{zip_clean[:5]}-{zip_clean[5:]}"
        return None


class AddressValidatorService:
    """
    Main address validation service.
    Uses RAG index for address matching and validation.
    """
    
    # Minimum similarity score to consider a match
    MIN_MATCH_SCORE = 0.75
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.9
    MEDIUM_CONFIDENCE = 0.7
    LOW_CONFIDENCE = 0.5
    
    def __init__(
        self,
        rag_indexer=None,
        embedding_client=None,
        opensearch_indexer=None
    ):
        self.rag_indexer = rag_indexer
        self.embedder = embedding_client
        self.opensearch = opensearch_indexer
    
    async def validate(self, address: dict) -> ValidationResult:
        """
        Validate an address against the RAG index and business rules.
        
        Args:
            address: Dict with keys like street, city, state, zip
            
        Returns:
            ValidationResult with is_valid flag, errors, and confidence score
        """
        errors = []
        context = {"checks_performed": []}
        
        # Step 1: Basic format validation
        format_errors = self._validate_format(address)
        errors.extend(format_errors)
        context["checks_performed"].append("format")
        
        # Step 2: State validation
        state_errors = self._validate_state(address)
        errors.extend(state_errors)
        context["checks_performed"].append("state")
        
        # Step 3: ZIP code validation
        zip_errors = self._validate_zip(address)
        errors.extend(zip_errors)
        context["checks_performed"].append("zip")
        
        # Step 4: Cross-field validation (ZIP-State, ZIP-City)
        cross_errors = await self._validate_cross_fields(address)
        errors.extend(cross_errors)
        context["checks_performed"].append("cross_field")
        
        # Step 5: RAG index lookup
        rag_result = await self._validate_against_rag(address)
        context["rag_match"] = rag_result
        context["checks_performed"].append("rag_index")
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            format_valid=len(format_errors) == 0,
            state_valid=len(state_errors) == 0,
            zip_valid=len(zip_errors) == 0,
            cross_valid=len(cross_errors) == 0,
            rag_score=rag_result.get("score", 0) if rag_result else 0
        )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            confidence=confidence,
            context=context,
            matched_address=rag_result.get("matched") if rag_result else None
        )
    
    def _validate_format(self, address: dict) -> list[ValidationError]:
        """Validate basic address format"""
        errors = []
        
        # Check required fields
        required = ["street", "city", "state", "zip"]
        for field_name in required:
            value = address.get(field_name) or address.get(field_name.upper())
            if not value or not str(value).strip():
                errors.append(ValidationError(
                    error_type=ValidationErrorType.MISSING_REQUIRED,
                    field=field_name,
                    message=f"Missing required field: {field_name}"
                ))
        
        # Street format check
        street = address.get("street") or address.get("address")
        if street:
            # Should have a number at the start (common pattern)
            if not re.match(r"^\d+\s+", str(street).strip()):
                # Not an error, just a note - many valid addresses don't start with numbers
                pass
        
        return errors
    
    def _validate_state(self, address: dict) -> list[ValidationError]:
        """Validate state field"""
        errors = []
        state = address.get("state") or address.get("st")
        
        if not state:
            return errors  # Already caught in format validation
        
        if not USStateValidator.is_valid_state(state):
            similar = USStateValidator.get_similar_states(state)
            suggestion = similar[0][0] if similar else None
            
            errors.append(ValidationError(
                error_type=ValidationErrorType.INVALID_STATE,
                field="state",
                message=f"Invalid state: '{state}'",
                original_value=state,
                suggestion=suggestion
            ))
        
        return errors
    
    def _validate_zip(self, address: dict) -> list[ValidationError]:
        """Validate ZIP code field"""
        errors = []
        zip_code = address.get("zip") or address.get("zipcode") or address.get("postal_code")
        
        if not zip_code:
            return errors  # Already caught in format validation
        
        if not ZipCodeValidator.is_valid_format(str(zip_code)):
            errors.append(ValidationError(
                error_type=ValidationErrorType.INVALID_ZIP,
                field="zip",
                message=f"Invalid ZIP code format: '{zip_code}'",
                original_value=zip_code
            ))
        
        return errors
    
    async def _validate_cross_fields(self, address: dict) -> list[ValidationError]:
        """Validate relationships between fields (ZIP-State, ZIP-City)"""
        errors = []
        
        zip_code = address.get("zip") or address.get("zipcode")
        state = address.get("state")
        city = address.get("city")
        
        if not zip_code or not state:
            return errors
        
        zip5 = ZipCodeValidator.extract_zip5(str(zip_code))
        normalized_state = USStateValidator.normalize_state(str(state))
        
        if zip5 and normalized_state:
            # Check ZIP-State match using first digit ranges
            # This is a simplified check - in production use USPS data
            zip_state_ranges = self._get_zip_state_ranges()
            first_digit = zip5[0]
            
            expected_states = zip_state_ranges.get(first_digit, [])
            if expected_states and normalized_state not in expected_states:
                errors.append(ValidationError(
                    error_type=ValidationErrorType.ZIP_STATE_MISMATCH,
                    field="zip",
                    message=f"ZIP code {zip5} doesn't match state {normalized_state}",
                    original_value=zip5,
                    suggestion=f"Expected states for ZIP starting with {first_digit}: {', '.join(expected_states[:5])}"
                ))
        
        return errors
    
    async def _validate_against_rag(self, address: dict) -> Optional[dict]:
        """Validate address against RAG index"""
        if not self.embedder or not self.opensearch:
            return None
        
        try:
            # Generate embedding for input address
            address_text = self._format_address_text(address)
            embedding = await self.embedder.get_embedding(address_text)
            
            # Search for similar addresses
            matches = await self.opensearch.search_similar_addresses(
                embedding=embedding,
                k=5,
                min_score=self.MIN_MATCH_SCORE
            )
            
            if matches:
                best_match = matches[0]
                return {
                    "found": True,
                    "score": best_match["score"],
                    "matched": {
                        "full_address": best_match["full_address"],
                        "street": best_match["street"],
                        "city": best_match["city"],
                        "state": best_match["state"],
                        "zip": best_match["zip"]
                    },
                    "alternatives": matches[1:4] if len(matches) > 1 else []
                }
            
            return {"found": False, "score": 0}
            
        except Exception as e:
            logger.error(f"RAG validation error: {e}")
            return None
    
    def _calculate_confidence(
        self,
        format_valid: bool,
        state_valid: bool,
        zip_valid: bool,
        cross_valid: bool,
        rag_score: float
    ) -> float:
        """Calculate overall validation confidence score"""
        # Weight each component
        weights = {
            "format": 0.15,
            "state": 0.15,
            "zip": 0.15,
            "cross": 0.20,
            "rag": 0.35
        }
        
        score = 0.0
        
        if format_valid:
            score += weights["format"]
        if state_valid:
            score += weights["state"]
        if zip_valid:
            score += weights["zip"]
        if cross_valid:
            score += weights["cross"]
        
        # RAG score contributes proportionally
        score += weights["rag"] * min(rag_score, 1.0)
        
        return round(score, 3)
    
    def _format_address_text(self, address: dict) -> str:
        """Format address for embedding"""
        parts = []
        for field_name in ["street", "address", "street2", "city", "state", "zip"]:
            value = address.get(field_name)
            if value:
                parts.append(str(value))
        return " ".join(parts)
    
    def _get_zip_state_ranges(self) -> dict:
        """
        Get ZIP code first-digit to state mappings.
        This is simplified - production should use full USPS data.
        """
        return {
            "0": ["CT", "MA", "ME", "NH", "NJ", "NY", "PR", "RI", "VT", "VI"],
            "1": ["DE", "NY", "PA"],
            "2": ["DC", "MD", "NC", "SC", "VA", "WV"],
            "3": ["AL", "FL", "GA", "MS", "TN"],
            "4": ["IN", "KY", "MI", "OH"],
            "5": ["IA", "MN", "MT", "ND", "SD", "WI"],
            "6": ["IL", "KS", "MO", "NE"],
            "7": ["AR", "LA", "OK", "TX"],
            "8": ["AZ", "CO", "ID", "NM", "NV", "UT", "WY"],
            "9": ["AK", "CA", "HI", "OR", "WA"]
        }
