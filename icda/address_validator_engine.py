"""Comprehensive address validation engine with confidence scoring.

This module provides a production-grade address validation system that:
- Validates addresses against known standards (USPS)
- Completes partial addresses intelligently
- Corrects common errors and typos
- Provides detailed confidence scores with component breakdown
- Suggests fixes for invalid addresses

The engine operates in multiple modes:
- VALIDATE: Check if address is valid and deliverable
- COMPLETE: Fill in missing components
- CORRECT: Fix errors while preserving intent
- STANDARDIZE: Format to USPS standard

Enhanced with:
- Unified confidence thresholds from config
- Robust error handling
- Input sanitization
- Spanish typo corrections for PR addresses
"""

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any

from icda.address_models import (
    AddressComponent,
    AddressQuality,
    ParsedAddress,
    VerificationStatus,
)
from icda.address_normalizer import (
    AddressNormalizer,
    STATE_ABBREVIATIONS,
    STREET_TYPES,
    is_puerto_rico_zip,
)
from icda.config import cfg
from icda.utils.resilience import sanitize_address_input

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Constants
# =============================================================================

# ZIP code to city/state mappings for completion (subset - would be full USPS DB in production)
_CITY_NEW_YORK = "New York"
_CITY_SAN_FRANCISCO = "San Francisco"
_CITY_SAN_JUAN = "San Juan"

ZIP_CITY_STATE: dict[str, tuple[str, str]] = {
    # Virginia
    "22201": ("Arlington", "VA"),
    "22202": ("Arlington", "VA"),
    "22203": ("Arlington", "VA"),
    "22222": ("Springfield", "VA"),
    "22150": ("Springfield", "VA"),
    "22151": ("Springfield", "VA"),
    "22030": ("Fairfax", "VA"),
    "22031": ("Fairfax", "VA"),
    "22041": ("Falls Church", "VA"),
    "22042": ("Falls Church", "VA"),
    "22101": ("McLean", "VA"),
    "22102": ("McLean", "VA"),
    "22180": ("Vienna", "VA"),
    "22181": ("Vienna", "VA"),
    "22182": ("Vienna", "VA"),
    "20001": ("Washington", "DC"),
    "20002": ("Washington", "DC"),
    "20003": ("Washington", "DC"),
    # New York
    "10001": (_CITY_NEW_YORK, "NY"),
    "10002": (_CITY_NEW_YORK, "NY"),
    "10003": (_CITY_NEW_YORK, "NY"),
    "10004": (_CITY_NEW_YORK, "NY"),
    "10005": (_CITY_NEW_YORK, "NY"),
    "10010": (_CITY_NEW_YORK, "NY"),
    "10011": (_CITY_NEW_YORK, "NY"),
    "10012": (_CITY_NEW_YORK, "NY"),
    "10013": (_CITY_NEW_YORK, "NY"),
    "10014": (_CITY_NEW_YORK, "NY"),
    "11201": ("Brooklyn", "NY"),
    "11202": ("Brooklyn", "NY"),
    "11203": ("Brooklyn", "NY"),
    # California
    "90001": ("Los Angeles", "CA"),
    "90002": ("Los Angeles", "CA"),
    "90210": ("Beverly Hills", "CA"),
    "94102": (_CITY_SAN_FRANCISCO, "CA"),
    "94103": (_CITY_SAN_FRANCISCO, "CA"),
    "94104": (_CITY_SAN_FRANCISCO, "CA"),
    "92101": ("San Diego", "CA"),
    "92102": ("San Diego", "CA"),
    # Texas
    "75201": ("Dallas", "TX"),
    "75202": ("Dallas", "TX"),
    "77001": ("Houston", "TX"),
    "77002": ("Houston", "TX"),
    "78201": ("San Antonio", "TX"),
    "78202": ("San Antonio", "TX"),
    "73301": ("Austin", "TX"),
    # Florida
    "33101": ("Miami", "FL"),
    "33102": ("Miami", "FL"),
    "32801": ("Orlando", "FL"),
    "32802": ("Orlando", "FL"),
    # Illinois
    "60601": ("Chicago", "IL"),
    "60602": ("Chicago", "IL"),
    "60603": ("Chicago", "IL"),
    # Puerto Rico
    "00901": (_CITY_SAN_JUAN, "PR"),
    "00902": (_CITY_SAN_JUAN, "PR"),
    "00907": (_CITY_SAN_JUAN, "PR"),
    "00926": ("Rio Piedras", "PR"),
    "00927": (_CITY_SAN_JUAN, "PR"),
    "00961": ("Bayamon", "PR"),
    "00983": ("Carolina", "PR"),
}

# Common address typos and corrections (English)
COMMON_TYPOS: dict[str, str] = {
    "stret": "street",
    "stree": "street",
    "steet": "street",
    "strret": "street",
    "avenu": "avenue",
    "avene": "avenue",
    "aveune": "avenue",
    "bulevard": "boulevard",
    "bolevard": "boulevard",
    "boulevar": "boulevard",
    "drve": "drive",
    "drvie": "drive",
    "lnae": "lane",
    "laen": "lane",
    "raod": "road",
    "roaad": "road",
    "circl": "circle",
    "cirle": "circle",
    "plac": "place",
    "palce": "place",
    "crout": "court",
    "coourt": "court",
    "apratment": "apartment",
    "apartmnet": "apartment",
    "appartment": "apartment",
    "suiet": "suite",
    "siute": "suite",
}

# Spanish typos common in Puerto Rico addresses
SPANISH_TYPOS: dict[str, str] = {
    # Urbanization typos
    "urbanizacion": "urbanizacion",
    "urbanisacion": "urbanizacion",
    "urbanizasion": "urbanizacion",
    "urbanzacion": "urbanizacion",
    "urbnizacion": "urbanizacion",
    # Calle (street) typos
    "cale": "calle",
    "callel": "calle",
    "callle": "calle",
    # Avenida typos
    "avenuda": "avenida",
    "avnida": "avenida",
    "avenid": "avenida",
    "avendida": "avenida",
    # Edificio typos
    "ediicio": "edificio",
    "edifcio": "edificio",
    "edficio": "edificio",
    # Apartamento typos
    "apartmento": "apartamento",
    "apartamneto": "apartamento",
    "apartameto": "apartamento",
    # Condominio typos
    "condiminio": "condominio",
    "condomino": "condominio",
    "condminio": "condominio",
    # Residencial typos
    "residencial": "residencial",
    "residensial": "residencial",
    "recidencial": "residencial",
    # Sector/Barrio typos
    "sectr": "sector",
    "sectro": "sector",
    "barri": "barrio",
    "bario": "barrio",
}

# State name typos
STATE_TYPOS: dict[str, str] = {
    "virgina": "virginia",
    "virignia": "virginia",
    "californa": "california",
    "californai": "california",
    "flordia": "florida",
    "floirda": "florida",
    "newyork": "new york",
    "texax": "texas",
    "texsa": "texas",
    "illinios": "illinois",
    "illnois": "illinois",
}


class ValidationMode(str, Enum):
    """Mode of validation operation."""

    VALIDATE = "validate"  # Check validity only
    COMPLETE = "complete"  # Fill missing components
    CORRECT = "correct"  # Fix errors
    STANDARDIZE = "standardize"  # Format to standard


class ComponentConfidence(str, Enum):
    """Confidence level for individual components."""

    EXACT = "exact"  # Exact match verified
    HIGH = "high"  # Very confident (> 90%)
    MEDIUM = "medium"  # Reasonably confident (70-90%)
    LOW = "low"  # Uncertain (50-70%)
    INFERRED = "inferred"  # Derived from other components
    MISSING = "missing"  # Component not present


@dataclass(slots=True)
class ComponentScore:
    """Detailed score for a single address component."""

    component: AddressComponent
    confidence: ComponentConfidence
    score: float  # 0.0 - 1.0
    original_value: str | None
    validated_value: str | None
    was_corrected: bool
    was_completed: bool
    correction_reason: str | None = None
    alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component.value,
            "confidence": self.confidence.value,
            "score": round(self.score, 4),
            "original_value": self.original_value,
            "validated_value": self.validated_value,
            "was_corrected": self.was_corrected,
            "was_completed": self.was_completed,
            "correction_reason": self.correction_reason,
            "alternatives": self.alternatives,
        }


@dataclass(slots=True)
class ValidationIssue:
    """Issue found during validation."""

    severity: str  # "error", "warning", "info"
    component: AddressComponent | None
    message: str
    suggestion: str | None = None
    auto_fixable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "component": self.component.value if self.component else None,
            "message": self.message,
            "suggestion": self.suggestion,
            "auto_fixable": self.auto_fixable,
        }


@dataclass(slots=True)
class ValidationResult:
    """Comprehensive validation result with detailed scoring."""

    # Overall status
    is_valid: bool
    is_deliverable: bool
    overall_confidence: float  # 0.0 - 1.0
    quality: AddressQuality
    status: VerificationStatus

    # Address data
    original: ParsedAddress
    validated: ParsedAddress | None
    standardized: str | None  # USPS-formatted single line

    # Component-level details
    component_scores: list[ComponentScore]
    issues: list[ValidationIssue]
    corrections_applied: list[str]
    completions_applied: list[str]

    # Alternatives
    alternatives: list[ParsedAddress]

    # Puerto Rico specific
    is_puerto_rico: bool = False
    urbanization_status: str | None = None  # "present", "missing", "inferred"

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "is_deliverable": self.is_deliverable,
            "overall_confidence": round(self.overall_confidence, 4),
            "confidence_percent": round(self.overall_confidence * 100, 1),
            "quality": self.quality.value,
            "status": self.status.value,
            "original": self.original.to_dict(),
            "validated": self.validated.to_dict() if self.validated else None,
            "standardized": self.standardized,
            "component_scores": [cs.to_dict() for cs in self.component_scores],
            "issues": [i.to_dict() for i in self.issues],
            "corrections_applied": self.corrections_applied,
            "completions_applied": self.completions_applied,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "is_puerto_rico": self.is_puerto_rico,
            "urbanization_status": self.urbanization_status,
            "metadata": self.metadata,
        }


class AddressValidatorEngine:
    """Comprehensive address validation engine.

    Provides multi-layered validation with:
    - Component-level scoring
    - Intelligent completion of partial addresses
    - Error correction with change tracking
    - Confidence calculation
    - Fix suggestions
    - Robust error handling for all operations
    """

    # Component weights for overall confidence calculation
    COMPONENT_WEIGHTS = {
        AddressComponent.STREET_NUMBER: 0.20,
        AddressComponent.STREET_NAME: 0.25,
        AddressComponent.STREET_TYPE: 0.05,
        AddressComponent.CITY: 0.15,
        AddressComponent.STATE: 0.15,
        AddressComponent.ZIP_CODE: 0.15,
        AddressComponent.UNIT: 0.03,
        AddressComponent.URBANIZATION: 0.02,  # Only for PR
    }

    def __init__(self, address_index=None):
        """Initialize the validator engine.

        Args:
            address_index: Optional AddressIndex for known address lookup.
        """
        self.index = address_index
        # Use unified thresholds from config
        self.THRESHOLD_VALID = cfg.address_threshold_valid
        self.THRESHOLD_DELIVERABLE = cfg.address_threshold_deliverable
        self.THRESHOLD_EXACT = cfg.address_threshold_exact
        self.MAX_ADDRESS_LENGTH = cfg.max_address_length
        self.MAX_TYPO_CORRECTIONS = cfg.max_typo_corrections

    def validate(
        self,
        raw_address: str,
        mode: ValidationMode = ValidationMode.CORRECT,
        context: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Validate an address with comprehensive scoring.

        Args:
            raw_address: Raw address string to validate.
            mode: Validation mode (validate, complete, correct, standardize).
            context: Optional context hints (e.g., known ZIP, state).

        Returns:
            ValidationResult with detailed scoring and corrections.
        """
        context = context or {}

        # Step 0: Sanitize input
        sanitized_address = sanitize_address_input(
            raw_address, max_length=self.MAX_ADDRESS_LENGTH
        )

        # Handle empty input
        if not sanitized_address:
            return self._create_invalid_result(
                raw_address or "", "Empty or invalid address input"
            )

        try:
            # Step 1: Pre-process and correct obvious typos
            corrected_input, typo_corrections = self._correct_typos(sanitized_address)

            # Step 2: Parse the address
            parsed = AddressNormalizer.normalize(corrected_input)

            # Step 3: Apply context hints
            parsed = self._apply_context(parsed, context)

            # Step 4: Complete missing components if possible
            completed, completions = self._complete_address(parsed, mode)

            # Step 5: Validate each component
            component_scores = self._score_components(parsed, completed)

            # Step 6: Identify issues
            issues = self._identify_issues(completed, component_scores)

            # Step 7: Calculate overall confidence
            overall_confidence = self._calculate_confidence(component_scores, completed)

            # Step 8: Determine quality and status
            quality = self._determine_quality(overall_confidence, issues)
            status = self._determine_status(
                overall_confidence, typo_corrections, completions
            )

            # Step 9: Check validity and deliverability
            is_valid = overall_confidence >= self.THRESHOLD_VALID and not any(
                i.severity == "error" for i in issues
            )
            is_deliverable = (
                overall_confidence >= self.THRESHOLD_DELIVERABLE and is_valid
            )

            # Step 10: Generate standardized format (even for partial/inferred results)
            standardized = None
            if completed.street_name or completed.city or completed.zip_code:
                try:
                    standardized = self._standardize_address(completed)
                except Exception as e:
                    logger.warning(f"Failed to standardize address: {e}")

            # Step 11: Find alternatives if needed
            alternatives = []
            if not is_valid and self.index:
                try:
                    alternatives = self._find_alternatives(parsed)
                except Exception as e:
                    logger.warning(f"Failed to find alternatives: {e}")

            # Step 12: Handle Puerto Rico specifics
            is_pr = completed.is_puerto_rico
            urb_status = None
            if is_pr:
                if completed.urbanization:
                    urb_status = "present"
                elif self._could_infer_urbanization(completed):
                    urb_status = "inferred"
                else:
                    urb_status = "missing"

            # Combine corrections
            all_corrections = typo_corrections + [
                cs.correction_reason
                for cs in component_scores
                if cs.was_corrected and cs.correction_reason
            ]

            return ValidationResult(
                is_valid=is_valid,
                is_deliverable=is_deliverable,
                overall_confidence=overall_confidence,
                quality=quality,
                status=status,
                original=parsed,
                # In COMPLETE mode, callers expect our best-effort completion even
                # if the address isn't fully "valid" yet. We return the completed
                # structure when confidence is at least "suggested" so the UI/API
                # can show the best guess along with component scores/issues.
                validated=(
                    completed
                    if (
                        is_valid
                        or (
                            mode == ValidationMode.COMPLETE
                            and overall_confidence >= 0.50
                        )
                    )
                    else None
                ),
                standardized=standardized,
                component_scores=component_scores,
                issues=issues,
                corrections_applied=all_corrections,
                completions_applied=completions,
                alternatives=alternatives,
                is_puerto_rico=is_pr,
                urbanization_status=urb_status,
                metadata={
                    "mode": mode.value,
                    "context_applied": bool(context),
                    "input_sanitized": raw_address != sanitized_address,
                },
            )
        except Exception as e:
            logger.error(f"Validation error for address '{raw_address[:50]}...': {e}")
            return self._create_invalid_result(
                raw_address, f"Validation error: {str(e)}"
            )

    def _create_invalid_result(
        self,
        raw_address: str,
        error_message: str,
    ) -> ValidationResult:
        """Create an invalid result for error cases."""
        parsed = ParsedAddress(raw=raw_address)
        return ValidationResult(
            is_valid=False,
            is_deliverable=False,
            overall_confidence=0.0,
            quality=AddressQuality.INVALID,
            status=VerificationStatus.FAILED,
            original=parsed,
            validated=None,
            standardized=None,
            component_scores=[],
            issues=[
                ValidationIssue(
                    severity="error",
                    component=None,
                    message=error_message,
                    auto_fixable=False,
                )
            ],
            corrections_applied=[],
            completions_applied=[],
            alternatives=[],
            is_puerto_rico=False,
            urbanization_status=None,
            metadata={"error": error_message},
        )

    def _correct_typos(self, raw: str) -> tuple[str, list[str]]:
        """Correct common typos in address.

        Includes English and Spanish typo corrections.
        Respects MAX_TYPO_CORRECTIONS limit to prevent runaway processing.

        Returns:
            Tuple of (corrected string, list of corrections made).
        """
        corrections: list[str] = []
        result: str = raw

        # Lowercase for matching
        lower: str = raw.lower()

        # Check for English street type typos
        for typo, correct in COMMON_TYPOS.items():
            if len(corrections) >= self.MAX_TYPO_CORRECTIONS:
                break
            pattern = r"\b" + re.escape(typo) + r"\b"
            if re.search(pattern, lower, re.IGNORECASE):
                result = re.sub(pattern, correct, result, flags=re.IGNORECASE)
                corrections.append(f"Corrected '{typo}' to '{correct}'")
                lower = result.lower()
                corrections += 1

        # Check for Spanish typos (common in PR addresses)
        for typo, correct in SPANISH_TYPOS.items():
            if len(corrections) >= self.MAX_TYPO_CORRECTIONS:
                break
            pattern = r"\b" + re.escape(typo) + r"\b"
            if re.search(pattern, lower, re.IGNORECASE):
                result = re.sub(pattern, correct, result, flags=re.IGNORECASE)
                corrections.append(f"Corrected Spanish '{typo}' to '{correct}'")
                lower = result.lower()
                corrections += 1

        # Check for state name typos
        for typo, correct in STATE_TYPOS.items():
            if len(corrections) >= self.MAX_TYPO_CORRECTIONS:
                break
            if typo in lower:
                result = result.replace(typo, correct)
                result = result.replace(typo.title(), correct.title())
                corrections.append(f"Corrected state '{typo}' to '{correct}'")
                lower = result.lower()
                corrections += 1

        # Fix common formatting issues (don't count as typo corrections)
        # Double spaces
        if "  " in result:
            result = re.sub(r"\s+", " ", result)
            corrections.append("Normalized whitespace")

        # Missing space after comma
        if re.search(r",\S", result):
            result = re.sub(r",(\S)", r", \1", result)
            corrections.append("Added space after comma")

        return result.strip(), corrections

    def _apply_context(
        self,
        parsed: ParsedAddress,
        context: dict[str, Any],
    ) -> ParsedAddress:
        """Apply context hints to parsed address."""
        # Apply ZIP code hint
        if not parsed.zip_code and "zip" in context:
            parsed.zip_code = str(context["zip"])[:5]
            if is_puerto_rico_zip(parsed.zip_code):
                parsed.is_puerto_rico = True
                parsed.state = "PR"

        # Apply state hint
        if not parsed.state and "state" in context:
            parsed.state = STATE_ABBREVIATIONS.get(
                context["state"].lower(), context["state"].upper()[:2]
            )

        # Apply city hint
        if not parsed.city and "city" in context:
            parsed.city = context["city"].title()

        return parsed

    def _complete_address(
        self,
        parsed: ParsedAddress,
        mode: ValidationMode,
    ) -> tuple[ParsedAddress, list[str]]:
        """Complete missing components if possible.

        Returns:
            Tuple of (completed address, list of completions).
        """
        completions = []

        if mode == ValidationMode.VALIDATE:
            # Don't complete in validate-only mode
            return parsed, []

        # Create a copy to modify
        completed = ParsedAddress(
            raw=parsed.raw,
            street_number=parsed.street_number,
            street_name=parsed.street_name,
            street_type=parsed.street_type,
            unit=parsed.unit,
            city=parsed.city,
            state=parsed.state,
            zip_code=parsed.zip_code,
            zip_plus4=parsed.zip_plus4,
            urbanization=parsed.urbanization,
            is_puerto_rico=parsed.is_puerto_rico,
            components_found=parsed.components_found.copy(),
            components_missing=parsed.components_missing.copy(),
        )

        # Complete city/state from ZIP
        if completed.zip_code and completed.zip_code in ZIP_CITY_STATE:
            city, state = ZIP_CITY_STATE[completed.zip_code]

            if not completed.city:
                completed.city = city
                completions.append(
                    f"Inferred city '{city}' from ZIP {completed.zip_code}"
                )
                if AddressComponent.CITY not in completed.components_found:
                    completed.components_found.append(AddressComponent.CITY)
                if AddressComponent.CITY in completed.components_missing:
                    completed.components_missing.remove(AddressComponent.CITY)

            if not completed.state:
                completed.state = state
                completions.append(
                    f"Inferred state '{state}' from ZIP {completed.zip_code}"
                )
                if AddressComponent.STATE not in completed.components_found:
                    completed.components_found.append(AddressComponent.STATE)
                if AddressComponent.STATE in completed.components_missing:
                    completed.components_missing.remove(AddressComponent.STATE)

        # Try to complete from index if available
        if self.index and completed.street_name and completed.zip_code:
            try:
                matches = self.index.lookup_street_in_zip(
                    completed.street_name,
                    completed.zip_code,
                    threshold=0.7,
                )
                if matches:
                    best = matches[0]

                    # Complete street type if missing
                    if not completed.street_type and best.address.parsed.street_type:
                        completed.street_type = best.address.parsed.street_type
                        completions.append(
                            f"Completed street type to '{completed.street_type}'"
                        )

                    # Complete street name if partial match
                    if best.address.parsed.street_name:
                        orig_name = completed.street_name.lower()
                        match_name = best.address.parsed.street_name.lower()
                        if orig_name != match_name and match_name.startswith(orig_name):
                            completed.street_name = best.address.parsed.street_name
                            completions.append(
                                f"Completed street name to '{completed.street_name}'"
                            )
            except Exception as e:
                logger.warning(f"Index lookup failed during completion: {e}")

        # Detect and set Puerto Rico flag
        if completed.zip_code and is_puerto_rico_zip(completed.zip_code):
            completed.is_puerto_rico = True
            if not completed.state:
                completed.state = "PR"
                completions.append("Inferred state 'PR' from ZIP code")

        return completed, completions

    def _score_components(
        self,
        original: ParsedAddress,
        validated: ParsedAddress,
    ) -> list[ComponentScore]:
        """Score each address component."""
        scores = []

        # Street Number
        scores.append(self._score_street_number(original, validated))

        # Street Name
        scores.append(self._score_street_name(original, validated))

        # Street Type
        scores.append(self._score_street_type(original, validated))

        # City
        scores.append(self._score_city(original, validated))

        # State
        scores.append(self._score_state(original, validated))

        # ZIP Code
        scores.append(self._score_zip(original, validated))

        # Unit (if present)
        if original.unit or validated.unit:
            scores.append(self._score_unit(original, validated))

        # Urbanization (if PR)
        if validated.is_puerto_rico:
            scores.append(self._score_urbanization(original, validated))

        return scores

    def _score_street_number(
        self,
        original: ParsedAddress,
        validated: ParsedAddress,
    ) -> ComponentScore:
        """Score street number component."""
        orig = original.street_number
        val = validated.street_number

        if not orig and not val:
            return ComponentScore(
                component=AddressComponent.STREET_NUMBER,
                confidence=ComponentConfidence.MISSING,
                score=0.0,
                original_value=None,
                validated_value=None,
                was_corrected=False,
                was_completed=False,
            )

        if not orig and val:
            return ComponentScore(
                component=AddressComponent.STREET_NUMBER,
                confidence=ComponentConfidence.INFERRED,
                score=0.6,
                original_value=None,
                validated_value=val,
                was_corrected=False,
                was_completed=True,
            )

        if orig == val:
            return ComponentScore(
                component=AddressComponent.STREET_NUMBER,
                confidence=ComponentConfidence.EXACT,
                score=1.0,
                original_value=orig,
                validated_value=val,
                was_corrected=False,
                was_completed=False,
            )

        # Check for similar numbers (e.g., "101" vs "101A")
        similarity = SequenceMatcher(None, orig or "", val or "").ratio()
        if similarity >= 0.8:
            return ComponentScore(
                component=AddressComponent.STREET_NUMBER,
                confidence=ComponentConfidence.HIGH,
                score=similarity,
                original_value=orig,
                validated_value=val,
                was_corrected=True,
                was_completed=False,
                correction_reason=f"Street number adjusted from '{orig}' to '{val}'",
            )

        return ComponentScore(
            component=AddressComponent.STREET_NUMBER,
            confidence=ComponentConfidence.LOW,
            score=similarity,
            original_value=orig,
            validated_value=val,
            was_corrected=True,
            was_completed=False,
            correction_reason=f"Street number changed from '{orig}' to '{val}'",
        )

    def _score_street_name(
        self,
        original: ParsedAddress,
        validated: ParsedAddress,
    ) -> ComponentScore:
        """Score street name component."""
        orig = (original.street_name or "").lower()
        val = (validated.street_name or "").lower()

        if not orig and not val:
            return ComponentScore(
                component=AddressComponent.STREET_NAME,
                confidence=ComponentConfidence.MISSING,
                score=0.0,
                original_value=None,
                validated_value=None,
                was_corrected=False,
                was_completed=False,
            )

        if not orig and val:
            return ComponentScore(
                component=AddressComponent.STREET_NAME,
                confidence=ComponentConfidence.INFERRED,
                score=0.5,
                original_value=None,
                validated_value=validated.street_name,
                was_corrected=False,
                was_completed=True,
            )

        # Exact match
        if orig == val:
            return ComponentScore(
                component=AddressComponent.STREET_NAME,
                confidence=ComponentConfidence.EXACT,
                score=1.0,
                original_value=original.street_name,
                validated_value=validated.street_name,
                was_corrected=False,
                was_completed=False,
            )

        # Check if validated is completion of original
        if val.startswith(orig):
            return ComponentScore(
                component=AddressComponent.STREET_NAME,
                confidence=ComponentConfidence.HIGH,
                score=0.95,
                original_value=original.street_name,
                validated_value=validated.street_name,
                was_corrected=False,
                was_completed=True,
            )

        # Fuzzy match
        similarity = SequenceMatcher(None, orig, val).ratio()
        if similarity >= 0.85:
            confidence = ComponentConfidence.HIGH
        elif similarity >= 0.70:
            confidence = ComponentConfidence.MEDIUM
        else:
            confidence = ComponentConfidence.LOW

        return ComponentScore(
            component=AddressComponent.STREET_NAME,
            confidence=confidence,
            score=similarity,
            original_value=original.street_name,
            validated_value=validated.street_name,
            was_corrected=orig != val,
            was_completed=False,
            correction_reason=(
                f"Street name '{original.street_name}' -> '{validated.street_name}'"
                if orig != val
                else None
            ),
        )

    def _score_street_type(
        self,
        original: ParsedAddress,
        validated: ParsedAddress,
    ) -> ComponentScore:
        """Score street type component."""
        orig = original.street_type
        val = validated.street_type

        if not orig and not val:
            return ComponentScore(
                component=AddressComponent.STREET_TYPE,
                confidence=ComponentConfidence.MISSING,
                score=0.3,  # Street type is often optional
                original_value=None,
                validated_value=None,
                was_corrected=False,
                was_completed=False,
            )

        if not orig and val:
            return ComponentScore(
                component=AddressComponent.STREET_TYPE,
                confidence=ComponentConfidence.INFERRED,
                score=0.8,
                original_value=None,
                validated_value=val,
                was_corrected=False,
                was_completed=True,
            )

        # Normalize both for comparison
        orig_norm = STREET_TYPES.get((orig or "").lower(), orig)
        val_norm = STREET_TYPES.get((val or "").lower(), val)

        if orig_norm == val_norm:
            return ComponentScore(
                component=AddressComponent.STREET_TYPE,
                confidence=ComponentConfidence.EXACT,
                score=1.0,
                original_value=orig,
                validated_value=val,
                was_corrected=False,
                was_completed=False,
            )

        return ComponentScore(
            component=AddressComponent.STREET_TYPE,
            confidence=ComponentConfidence.MEDIUM,
            score=0.7,
            original_value=orig,
            validated_value=val,
            was_corrected=True,
            was_completed=False,
            correction_reason=f"Street type standardized from '{orig}' to '{val}'",
        )

    def _score_city(
        self,
        original: ParsedAddress,
        validated: ParsedAddress,
    ) -> ComponentScore:
        """Score city component."""
        orig = (original.city or "").lower()
        val = (validated.city or "").lower()

        if not orig and not val:
            return ComponentScore(
                component=AddressComponent.CITY,
                confidence=ComponentConfidence.MISSING,
                score=0.0,
                original_value=None,
                validated_value=None,
                was_corrected=False,
                was_completed=False,
            )

        if not orig and val:
            return ComponentScore(
                component=AddressComponent.CITY,
                confidence=ComponentConfidence.INFERRED,
                score=0.85,  # High confidence if inferred from ZIP
                original_value=None,
                validated_value=validated.city,
                was_corrected=False,
                was_completed=True,
            )

        if orig == val:
            return ComponentScore(
                component=AddressComponent.CITY,
                confidence=ComponentConfidence.EXACT,
                score=1.0,
                original_value=original.city,
                validated_value=validated.city,
                was_corrected=False,
                was_completed=False,
            )

        # Fuzzy match for typos
        similarity = SequenceMatcher(None, orig, val).ratio()
        if similarity >= 0.85:
            confidence = ComponentConfidence.HIGH
        elif similarity >= 0.70:
            confidence = ComponentConfidence.MEDIUM
        else:
            confidence = ComponentConfidence.LOW

        return ComponentScore(
            component=AddressComponent.CITY,
            confidence=confidence,
            score=similarity,
            original_value=original.city,
            validated_value=validated.city,
            was_corrected=orig != val,
            was_completed=False,
            correction_reason=(
                f"City '{original.city}' -> '{validated.city}'" if orig != val else None
            ),
        )

    def _score_state(
        self,
        original: ParsedAddress,
        validated: ParsedAddress,
    ) -> ComponentScore:
        """Score state component."""
        orig = (original.state or "").upper()
        val = (validated.state or "").upper()

        if not orig and not val:
            return ComponentScore(
                component=AddressComponent.STATE,
                confidence=ComponentConfidence.MISSING,
                score=0.0,
                original_value=None,
                validated_value=None,
                was_corrected=False,
                was_completed=False,
            )

        if not orig and val:
            return ComponentScore(
                component=AddressComponent.STATE,
                confidence=ComponentConfidence.INFERRED,
                score=0.9,  # Very high confidence if from ZIP
                original_value=None,
                validated_value=val,
                was_corrected=False,
                was_completed=True,
            )

        if orig == val:
            return ComponentScore(
                component=AddressComponent.STATE,
                confidence=ComponentConfidence.EXACT,
                score=1.0,
                original_value=orig,
                validated_value=val,
                was_corrected=False,
                was_completed=False,
            )

        # State mismatch is serious
        return ComponentScore(
            component=AddressComponent.STATE,
            confidence=ComponentConfidence.LOW,
            score=0.0,
            original_value=orig,
            validated_value=val,
            was_corrected=True,
            was_completed=False,
            correction_reason=f"State changed from '{orig}' to '{val}' - verify correct",
        )

    def _score_zip(
        self,
        original: ParsedAddress,
        validated: ParsedAddress,
    ) -> ComponentScore:
        """Score ZIP code component."""
        orig = original.zip_code
        val = validated.zip_code

        if not orig and not val:
            return ComponentScore(
                component=AddressComponent.ZIP_CODE,
                confidence=ComponentConfidence.MISSING,
                score=0.0,
                original_value=None,
                validated_value=None,
                was_corrected=False,
                was_completed=False,
            )

        if not orig and val:
            return ComponentScore(
                component=AddressComponent.ZIP_CODE,
                confidence=ComponentConfidence.INFERRED,
                score=0.7,
                original_value=None,
                validated_value=val,
                was_corrected=False,
                was_completed=True,
            )

        if orig == val:
            return ComponentScore(
                component=AddressComponent.ZIP_CODE,
                confidence=ComponentConfidence.EXACT,
                score=1.0,
                original_value=orig,
                validated_value=val,
                was_corrected=False,
                was_completed=False,
            )

        # Check for transposition (common error)
        if sorted(orig or "") == sorted(val or ""):
            return ComponentScore(
                component=AddressComponent.ZIP_CODE,
                confidence=ComponentConfidence.MEDIUM,
                score=0.8,
                original_value=orig,
                validated_value=val,
                was_corrected=True,
                was_completed=False,
                correction_reason=f"ZIP transposition corrected: '{orig}' -> '{val}'",
            )

        # ZIP mismatch
        return ComponentScore(
            component=AddressComponent.ZIP_CODE,
            confidence=ComponentConfidence.LOW,
            score=0.3,
            original_value=orig,
            validated_value=val,
            was_corrected=True,
            was_completed=False,
            correction_reason=f"ZIP changed from '{orig}' to '{val}' - verify correct",
        )

    def _score_unit(
        self,
        original: ParsedAddress,
        validated: ParsedAddress,
    ) -> ComponentScore:
        """Score unit component."""
        orig = original.unit
        val = validated.unit

        if orig == val:
            score = 1.0 if orig else 0.5
            return ComponentScore(
                component=AddressComponent.UNIT,
                confidence=(
                    ComponentConfidence.EXACT if orig else ComponentConfidence.MISSING
                ),
                score=score,
                original_value=orig,
                validated_value=val,
                was_corrected=False,
                was_completed=False,
            )

        return ComponentScore(
            component=AddressComponent.UNIT,
            confidence=ComponentConfidence.MEDIUM,
            score=0.7,
            original_value=orig,
            validated_value=val,
            was_corrected=orig != val,
            was_completed=not orig and bool(val),
        )

    def _score_urbanization(
        self,
        original: ParsedAddress,
        validated: ParsedAddress,
    ) -> ComponentScore:
        """Score urbanization component (PR only)."""
        orig = original.urbanization
        val = validated.urbanization

        if not orig and not val:
            # Missing urbanization for PR is a warning
            return ComponentScore(
                component=AddressComponent.URBANIZATION,
                confidence=ComponentConfidence.MISSING,
                score=0.3,  # Low score - PR needs urbanization
                original_value=None,
                validated_value=None,
                was_corrected=False,
                was_completed=False,
            )

        if not orig and val:
            return ComponentScore(
                component=AddressComponent.URBANIZATION,
                confidence=ComponentConfidence.INFERRED,
                score=0.8,
                original_value=None,
                validated_value=val,
                was_corrected=False,
                was_completed=True,
            )

        if (orig or "").upper() == (val or "").upper():
            return ComponentScore(
                component=AddressComponent.URBANIZATION,
                confidence=ComponentConfidence.EXACT,
                score=1.0,
                original_value=orig,
                validated_value=val,
                was_corrected=False,
                was_completed=False,
            )

        similarity = SequenceMatcher(
            None, (orig or "").lower(), (val or "").lower()
        ).ratio()

        return ComponentScore(
            component=AddressComponent.URBANIZATION,
            confidence=(
                ComponentConfidence.MEDIUM
                if similarity > 0.7
                else ComponentConfidence.LOW
            ),
            score=similarity,
            original_value=orig,
            validated_value=val,
            was_corrected=True,
            was_completed=False,
            correction_reason=(
                f"Urbanization '{orig}' -> '{val}'" if orig != val else None
            ),
        )

    def _identify_issues(
        self,
        address: ParsedAddress,
        component_scores: list[ComponentScore],
    ) -> list[ValidationIssue]:
        """Identify validation issues."""
        issues = []

        # Check for missing required components
        if not address.street_number:
            issues.append(
                ValidationIssue(
                    severity="error",
                    component=AddressComponent.STREET_NUMBER,
                    message="Missing street number",
                    suggestion="Add street number (e.g., '123 Main St')",
                    auto_fixable=False,
                )
            )

        if not address.street_name:
            issues.append(
                ValidationIssue(
                    severity="error",
                    component=AddressComponent.STREET_NAME,
                    message="Missing street name",
                    suggestion="Add street name",
                    auto_fixable=False,
                )
            )

        if not address.city and not address.zip_code:
            issues.append(
                ValidationIssue(
                    severity="error",
                    component=AddressComponent.CITY,
                    message="Missing city and ZIP code - cannot determine location",
                    suggestion="Add at least city or ZIP code",
                    auto_fixable=False,
                )
            )

        if not address.state and not address.zip_code:
            issues.append(
                ValidationIssue(
                    severity="error",
                    component=AddressComponent.STATE,
                    message="Missing state and ZIP code - cannot determine location",
                    suggestion="Add at least state or ZIP code",
                    auto_fixable=False,
                )
            )

        # Warnings for optional but recommended
        if not address.zip_code:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    component=AddressComponent.ZIP_CODE,
                    message="Missing ZIP code - reduces delivery accuracy",
                    suggestion="Add 5-digit ZIP code",
                    auto_fixable=True,  # Can be inferred from city/state
                )
            )

        if not address.street_type:
            issues.append(
                ValidationIssue(
                    severity="info",
                    component=AddressComponent.STREET_TYPE,
                    message="Missing street type (St, Ave, Blvd, etc.)",
                    suggestion="Add street type suffix for clarity",
                    auto_fixable=True,
                )
            )

        # Puerto Rico specific
        if address.is_puerto_rico and not address.urbanization:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    component=AddressComponent.URBANIZATION,
                    message="Puerto Rico address missing urbanization (URB)",
                    suggestion="Add 'URB [name]' before street address for PR deliverability",
                    auto_fixable=False,
                )
            )

        # Check for low confidence components
        for score in component_scores:
            if score.confidence == ComponentConfidence.LOW and score.original_value:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        component=score.component,
                        message=f"Low confidence for {score.component.value}: '{score.original_value}'",
                        suggestion=f"Verify {score.component.value} is correct",
                        auto_fixable=False,
                    )
                )

        return issues

    def _calculate_confidence(
        self,
        component_scores: list[ComponentScore],
        address: ParsedAddress,
    ) -> float:
        """Calculate overall confidence score."""
        weighted_sum = 0.0
        total_weight = 0.0

        for score in component_scores:
            weight = self.COMPONENT_WEIGHTS.get(score.component, 0.05)

            # Adjust weights for PR addresses
            if (
                address.is_puerto_rico
                and score.component == AddressComponent.URBANIZATION
            ):
                weight = 0.10  # Higher weight for PR

            weighted_sum += score.score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        base_confidence = weighted_sum / total_weight

        # Apply bonuses/penalties
        # Bonus for complete addresses
        essential_present = (
            address.street_number
            and address.street_name
            and address.city
            and address.state
            and address.zip_code
        )
        if essential_present:
            base_confidence = min(1.0, base_confidence * 1.05)

        # Penalty for PR without urbanization
        if address.is_puerto_rico and not address.urbanization:
            base_confidence *= 0.85

        return min(1.0, max(0.0, base_confidence))

    def _determine_quality(
        self,
        confidence: float,
        issues: list[ValidationIssue],
    ) -> AddressQuality:
        """Determine address quality classification."""
        error_count = sum(1 for i in issues if i.severity == "error")
        warning_count = sum(1 for i in issues if i.severity == "warning")

        if error_count > 0:
            return AddressQuality.INVALID

        if confidence >= 0.95 and warning_count == 0:
            return AddressQuality.COMPLETE

        if confidence >= 0.70:
            return AddressQuality.PARTIAL

        if warning_count > 2:
            return AddressQuality.AMBIGUOUS

        return AddressQuality.PARTIAL

    def _determine_status(
        self,
        confidence: float,
        corrections: list[str],
        completions: list[str],
    ) -> VerificationStatus:
        """Determine verification status."""
        if confidence >= self.THRESHOLD_EXACT:
            return VerificationStatus.VERIFIED

        if corrections and confidence >= self.THRESHOLD_VALID:
            return VerificationStatus.CORRECTED

        if completions and confidence >= self.THRESHOLD_VALID:
            return VerificationStatus.COMPLETED

        if confidence >= 0.50:
            return VerificationStatus.SUGGESTED

        return VerificationStatus.UNVERIFIED

    def _standardize_address(self, address: ParsedAddress) -> str:
        """Format address to USPS standard single line."""
        parts = []

        # Urbanization line (PR)
        if address.urbanization and address.is_puerto_rico:
            parts.append(f"URB {address.urbanization.upper()}")

        # Street line
        street_parts = []
        if address.street_number:
            street_parts.append(address.street_number.upper())
        if address.street_name:
            street_parts.append(address.street_name.upper())
        if address.street_type:
            street_parts.append(address.street_type.upper())
        if street_parts:
            parts.append(" ".join(street_parts))

        # Unit
        if address.unit:
            parts.append(address.unit.upper())

        # City, State ZIP
        csz_parts = []
        if address.city:
            csz_parts.append(address.city.upper())
        if address.state:
            csz_parts.append(address.state.upper())
        if address.zip_code:
            zip_str = address.zip_code
            if address.zip_plus4:
                zip_str += f"-{address.zip_plus4}"
            csz_parts.append(zip_str)

        if csz_parts:
            # Format: CITY, STATE ZIP
            if len(csz_parts) >= 2:
                city_part = csz_parts[0] + ","
                parts.append(f"{city_part} {' '.join(csz_parts[1:])}")
            else:
                parts.append(" ".join(csz_parts))

        return ", ".join(parts)

    def _find_alternatives(
        self,
        address: ParsedAddress,
        limit: int = 5,
    ) -> list[ParsedAddress]:
        """Find alternative address matches."""
        if not self.index:
            return []

        alternatives = []

        # Try fuzzy matching
        matches = self.index.lookup_fuzzy(address, threshold=0.5)
        for match in matches[:limit]:
            alternatives.append(match.address.parsed)

        return alternatives

    def _could_infer_urbanization(self, address: ParsedAddress) -> bool:
        """Check if urbanization could be inferred for PR address."""
        if not self.index or not address.is_puerto_rico:
            return False

        # Try to find matches that include urbanization
        if address.zip_code:
            zip_matches = self.index.lookup_by_zip(address.zip_code)
            for match in zip_matches:
                if match.parsed.urbanization:
                    return True

        return False


# Convenience function for quick validation
def validate_address(
    address: str,
    mode: str = "correct",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Quick validation function.

    Args:
        address: Raw address string.
        mode: "validate", "complete", "correct", or "standardize".
        context: Optional context hints.

    Returns:
        Validation result as dictionary.
    """
    engine = AddressValidatorEngine()
    mode_enum = ValidationMode(mode)
    result = engine.validate(address, mode_enum, context)
    return result.to_dict()
