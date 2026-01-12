"""
ICDA Nova Pro Address Corrector
================================
Uses Amazon Nova Pro to intelligently correct invalid addresses.
Returns suggested corrections with confidence scores and reasoning.

Author: Bishop Walker / Salt Water Coder
Project: ICDA Prototype
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CorrectionSource(str, Enum):
    """Source of correction suggestion"""
    RAG_INDEX = "rag_index"          # From indexed addresses
    NOVA_INFERENCE = "nova_inference"  # Nova Pro reasoning
    FUZZY_MATCH = "fuzzy_match"       # String similarity matching
    ENSEMBLE = "ensemble"              # Combined from multiple sources


@dataclass
class AddressCorrection:
    """A single address correction suggestion"""
    corrected_address: dict
    confidence: float
    source: CorrectionSource
    reasoning: str
    changes_made: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "corrected_address": self.corrected_address,
            "confidence": round(self.confidence, 3),
            "source": self.source.value,
            "reasoning": self.reasoning,
            "changes_made": self.changes_made
        }


@dataclass
class CorrectionResult:
    """Result of correction attempt"""
    original_address: dict
    suggestions: list[dict]
    best_match: Optional[dict] = None
    processing_time_ms: int = 0
    
    def to_dict(self) -> dict:
        return {
            "original_address": self.original_address,
            "suggestions": self.suggestions,
            "best_match": self.best_match,
            "processing_time_ms": self.processing_time_ms
        }


class NovaProClient:
    """
    Client for Amazon Nova Pro model.
    Handles address correction reasoning.
    """
    
    MODEL_ID = "amazon.nova-pro-v1:0"
    
    def __init__(self, bedrock_client=None):
        self.client = bedrock_client
        self._initialized = False
    
    async def initialize(self):
        """Lazy initialization"""
        if self._initialized:
            return
        
        if not self.client:
            import boto3
            self.client = boto3.client(
                "bedrock-runtime",
                region_name="us-east-1"
            )
        self._initialized = True
    
    async def complete(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.2
    ) -> str:
        """Send completion request to Nova Pro"""
        await self.initialize()
        
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        
        request_body = {
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature
            }
        }
        
        if system_prompt:
            request_body["system"] = [{"text": system_prompt}]
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.invoke_model(
                    modelId=self.MODEL_ID,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(request_body)
                )
            )
            
            result = json.loads(response["body"].read())
            return result["output"]["message"]["content"][0]["text"]
            
        except Exception as e:
            logger.error(f"Nova Pro error: {e}")
            raise


class NovaProAddressCorrector:
    """
    Uses Nova Pro to suggest address corrections.
    Combines RAG index matching with LLM reasoning.
    """
    
    # Correction system prompt
    SYSTEM_PROMPT = """You are an expert US address validation and correction system.
Your task is to analyze invalid addresses and suggest corrections.

When analyzing an address:
1. Identify specific problems (typos, missing info, wrong format)
2. Use context clues to infer correct values
3. Consider common abbreviations and variants
4. Check geographic plausibility (ZIP matches state/city)

Always respond with valid JSON in this exact format:
{
    "corrections": [
        {
            "street": "corrected street",
            "city": "corrected city",
            "state": "corrected state (2-letter code)",
            "zip": "corrected zip",
            "confidence": 0.0 to 1.0,
            "reasoning": "why this correction",
            "changes": ["list of changes made"]
        }
    ],
    "analysis": "brief analysis of the problems found"
}

Provide up to 3 suggestions ranked by confidence.
If you cannot determine a correction, use confidence 0.0."""

    def __init__(
        self,
        nova_client: NovaProClient = None,
        embedding_client=None,
        opensearch_indexer=None
    ):
        self.nova = nova_client or NovaProClient()
        self.embedder = embedding_client
        self.opensearch = opensearch_indexer
    
    async def suggest_corrections(
        self,
        original_address: dict,
        validation_context: dict = None
    ) -> CorrectionResult:
        """
        Generate correction suggestions for an invalid address.
        
        Uses a multi-stage approach:
        1. RAG index lookup for similar addresses
        2. Nova Pro reasoning for intelligent corrections
        3. Fuzzy matching for string-level corrections
        4. Ensemble to combine and rank suggestions
        """
        import time
        start_time = time.time()
        
        all_suggestions = []
        
        # Stage 1: RAG index lookup
        rag_suggestions = await self._get_rag_suggestions(original_address)
        all_suggestions.extend(rag_suggestions)
        
        # Stage 2: Nova Pro reasoning
        nova_suggestions = await self._get_nova_suggestions(
            original_address,
            validation_context,
            rag_suggestions
        )
        all_suggestions.extend(nova_suggestions)
        
        # Stage 3: Fuzzy matching
        fuzzy_suggestions = self._get_fuzzy_suggestions(original_address)
        all_suggestions.extend(fuzzy_suggestions)
        
        # Stage 4: Ensemble - dedupe and rank
        final_suggestions = self._ensemble_suggestions(all_suggestions)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return CorrectionResult(
            original_address=original_address,
            suggestions=[s.to_dict() for s in final_suggestions[:5]],
            best_match=final_suggestions[0].to_dict() if final_suggestions else None,
            processing_time_ms=processing_time
        )
    
    async def _get_rag_suggestions(
        self,
        address: dict
    ) -> list[AddressCorrection]:
        """Get correction suggestions from RAG index"""
        suggestions = []
        
        if not self.embedder or not self.opensearch:
            return suggestions
        
        try:
            # Search for similar addresses
            address_text = self._format_address(address)
            embedding = await self.embedder.get_embedding(address_text)
            
            matches = await self.opensearch.search_similar_addresses(
                embedding=embedding,
                k=5,
                min_score=0.65
            )
            
            for match in matches:
                corrected = {
                    "street": match.get("street", ""),
                    "city": match.get("city", ""),
                    "state": match.get("state", ""),
                    "zip": match.get("zip", "")
                }
                
                changes = self._identify_changes(address, corrected)
                
                suggestions.append(AddressCorrection(
                    corrected_address=corrected,
                    confidence=match.get("score", 0.7),
                    source=CorrectionSource.RAG_INDEX,
                    reasoning=f"Found similar address in index with {match.get('score', 0):.1%} similarity",
                    changes_made=changes
                ))
                
        except Exception as e:
            logger.error(f"RAG suggestion error: {e}")
        
        return suggestions
    
    async def _get_nova_suggestions(
        self,
        address: dict,
        validation_context: dict,
        rag_suggestions: list[AddressCorrection]
    ) -> list[AddressCorrection]:
        """Get correction suggestions from Nova Pro"""
        suggestions = []
        
        try:
            prompt = self._build_correction_prompt(address, validation_context, rag_suggestions)
            response = await self.nova.complete(prompt, self.SYSTEM_PROMPT)
            suggestions = self._parse_nova_response(response, address)
        except Exception as e:
            logger.error(f"Nova suggestion error: {e}")
        
        return suggestions
    
    def _get_fuzzy_suggestions(
        self,
        address: dict
    ) -> list[AddressCorrection]:
        """Get correction suggestions from fuzzy matching"""
        suggestions = []
        
        corrected = dict(address)
        changes = []
        
        # Street corrections
        if address.get("street"):
            corrected_street = self._correct_street_typos(address["street"])
            if corrected_street != address["street"]:
                corrected["street"] = corrected_street
                changes.append(f"Street: '{address['street']}' → '{corrected_street}'")
        
        # State normalization
        if address.get("state"):
            from .address_validator import USStateValidator
            normalized = USStateValidator.normalize_state(address["state"])
            if normalized and normalized != address["state"].upper():
                corrected["state"] = normalized
                changes.append(f"State: '{address['state']}' → '{normalized}'")
        
        # ZIP normalization
        if address.get("zip"):
            from .address_validator import ZipCodeValidator
            normalized = ZipCodeValidator.normalize(str(address["zip"]))
            if normalized and normalized != address["zip"]:
                corrected["zip"] = normalized
                changes.append(f"ZIP: '{address['zip']}' → '{normalized}'")
        
        if changes:
            suggestions.append(AddressCorrection(
                corrected_address=corrected,
                confidence=0.6,
                source=CorrectionSource.FUZZY_MATCH,
                reasoning="Applied common typo corrections and normalizations",
                changes_made=changes
            ))
        
        return suggestions
    
    def _ensemble_suggestions(
        self,
        all_suggestions: list[AddressCorrection]
    ) -> list[AddressCorrection]:
        """Combine and rank all suggestions"""
        if not all_suggestions:
            return []
        
        # Group by corrected address (dedupe)
        grouped = {}
        for suggestion in all_suggestions:
            key = self._address_key(suggestion.corrected_address)
            
            if key in grouped:
                # Combine confidence scores
                existing = grouped[key]
                # Boost confidence when multiple sources agree
                combined_confidence = min(
                    1.0,
                    existing.confidence * 0.6 + suggestion.confidence * 0.5
                )
                
                if suggestion.confidence > existing.confidence:
                    grouped[key] = AddressCorrection(
                        corrected_address=suggestion.corrected_address,
                        confidence=combined_confidence,
                        source=CorrectionSource.ENSEMBLE,
                        reasoning=f"Multiple sources agree: {existing.source.value}, {suggestion.source.value}",
                        changes_made=suggestion.changes_made
                    )
                else:
                    existing.confidence = combined_confidence
                    existing.source = CorrectionSource.ENSEMBLE
                    existing.reasoning = f"Multiple sources agree: {existing.source.value}, {suggestion.source.value}"
            else:
                grouped[key] = suggestion
        
        # Sort by confidence
        sorted_suggestions = sorted(
            grouped.values(),
            key=lambda x: x.confidence,
            reverse=True
        )
        
        return sorted_suggestions
    
    def _build_correction_prompt(
        self,
        address: dict,
        validation_context: dict,
        rag_suggestions: list[AddressCorrection]
    ) -> str:
        """Build prompt for Nova Pro correction"""
        prompt_parts = [
            "Please analyze this invalid address and suggest corrections:\n",
            f"ORIGINAL ADDRESS:",
            f"  Street: {address.get('street', 'N/A')}",
            f"  City: {address.get('city', 'N/A')}",
            f"  State: {address.get('state', 'N/A')}",
            f"  ZIP: {address.get('zip', 'N/A')}",
            ""
        ]
        
        if validation_context:
            errors = validation_context.get("errors", [])
            if errors:
                prompt_parts.append("VALIDATION ERRORS:")
                for error in errors[:5]:
                    if isinstance(error, dict):
                        prompt_parts.append(f"  - {error.get('message', str(error))}")
                    else:
                        prompt_parts.append(f"  - {error}")
                prompt_parts.append("")
        
        if rag_suggestions:
            prompt_parts.append("SIMILAR ADDRESSES IN DATABASE:")
            for i, sug in enumerate(rag_suggestions[:3], 1):
                addr = sug.corrected_address
                prompt_parts.append(
                    f"  {i}. {addr.get('street', '')}, {addr.get('city', '')}, "
                    f"{addr.get('state', '')} {addr.get('zip', '')} "
                    f"(confidence: {sug.confidence:.1%})"
                )
            prompt_parts.append("")
        
        prompt_parts.append(
            "Based on the above, provide your correction suggestions in JSON format."
        )
        
        return "\n".join(prompt_parts)
    
    def _parse_nova_response(
        self,
        response: str,
        original_address: dict
    ) -> list[AddressCorrection]:
        """Parse Nova Pro response into corrections"""
        suggestions = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if not json_match:
                return []
            
            data = json.loads(json_match.group())
            corrections = data.get("corrections", [])
            
            for corr in corrections:
                if not isinstance(corr, dict):
                    continue
                
                confidence = float(corr.get("confidence", 0))
                if confidence < 0.1:
                    continue
                
                corrected = {
                    "street": corr.get("street", original_address.get("street")),
                    "city": corr.get("city", original_address.get("city")),
                    "state": corr.get("state", original_address.get("state")),
                    "zip": corr.get("zip", original_address.get("zip"))
                }
                
                changes = corr.get("changes", [])
                if not changes:
                    changes = self._identify_changes(original_address, corrected)
                
                suggestions.append(AddressCorrection(
                    corrected_address=corrected,
                    confidence=confidence,
                    source=CorrectionSource.NOVA_INFERENCE,
                    reasoning=corr.get("reasoning", "Nova Pro correction"),
                    changes_made=changes
                ))
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Error parsing Nova response: {e}")
        
        return suggestions
    
    def _identify_changes(
        self,
        original: dict,
        corrected: dict
    ) -> list[str]:
        """Identify what changed between original and corrected"""
        changes = []
        
        for field_name in ["street", "city", "state", "zip"]:
            orig_val = str(original.get(field_name, "")).strip().lower()
            corr_val = str(corrected.get(field_name, "")).strip().lower()
            
            if orig_val != corr_val and corr_val:
                changes.append(f"{field_name}: '{original.get(field_name, '')}' → '{corrected.get(field_name, '')}'")
        
        return changes
    
    def _format_address(self, address: dict) -> str:
        """Format address dict into string"""
        parts = []
        for field_name in ["street", "city", "state", "zip"]:
            value = address.get(field_name)
            if value:
                parts.append(str(value))
        return " ".join(parts)
    
    def _address_key(self, address: dict) -> str:
        """Generate unique key for address (for deduplication)"""
        parts = [
            str(address.get("street", "")).lower().strip(),
            str(address.get("city", "")).lower().strip(),
            str(address.get("state", "")).upper().strip(),
            str(address.get("zip", ""))[:5].strip()
        ]
        return "|".join(parts)
    
    def _correct_street_typos(self, street: str) -> str:
        """Apply common street typo corrections"""
        corrections = {
            # Common typos
            r"\bturkey\b": "Turkey Run",  # Your 101 turkey 22030 example!
            r"\bst\.?\b": "St",
            r"\bave\.?\b": "Ave",
            r"\bblvd\.?\b": "Blvd",
            r"\brd\.?\b": "Rd",
            r"\bdr\.?\b": "Dr",
            r"\bln\.?\b": "Ln",
            r"\bct\.?\b": "Ct",
            r"\bcir\.?\b": "Cir",
            r"\bn\.?\s": "N ",
            r"\bs\.?\s": "S ",
            r"\be\.?\s": "E ",
            r"\bw\.?\s": "W ",
        }
        
        result = street
        for pattern, replacement in corrections.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
