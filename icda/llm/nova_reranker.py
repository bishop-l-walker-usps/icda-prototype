"""Nova-based reranker for address completion with few-shot learning.

Uses AWS Bedrock Nova to intelligently select and complete addresses
from vector search candidates.
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


@dataclass
class RerankerResult:
    """Result from Nova reranking."""
    match: Optional[str]  # Best matching full address
    confidence: float  # 0-1 confidence score
    reason: Optional[str] = None  # Explanation (if requested)
    crid: Optional[str] = None  # Customer ID if matched


# Few-shot examples for address completion
FEW_SHOT_EXAMPLES = """You are an address completion expert. Given a partial/messy address and candidate matches from a database, select the best match.

RULES:
1. Prioritize exact street number matches
2. Use fuzzy matching for street names (typos are common)
3. ZIP code match is strong signal
4. Consider phonetic similarity (e.g., "mane" = "main")
5. If no good match, return null

Example 1:
Input: "101 turkey 22222"
Candidates:
1. 101 Turkey Run, Springfield, VA 22222 (score: 0.89)
2. 101 Turkey Trot Ln, Springfield, VA 22222 (score: 0.85)
3. 1010 Turkey Creek Rd, Richmond, VA 22222 (score: 0.72)
Output: {"best_match": "101 Turkey Run, Springfield, VA 22222", "confidence": 0.95, "reason": "Exact street number, 'turkey' matches 'Turkey Run', exact ZIP"}

Example 2:
Input: "456 oak apt 3 chicago"
Candidates:
1. 456 Oak Ave Apt 3, Chicago, IL 60601 (score: 0.91)
2. 456 Oak St Apt 3, Chicago, IL 60602 (score: 0.88)
3. 456 Oakland Ave #3, Chicago, IL 60601 (score: 0.75)
Output: {"best_match": "456 Oak Ave Apt 3, Chicago, IL 60601", "confidence": 0.92, "reason": "All components match, Oak Ave more common than Oak St"}

Example 3:
Input: "789 mane street san fran"
Candidates:
1. 789 Main St, San Francisco, CA 94102 (score: 0.82)
2. 789 Maine Ave, San Francisco, CA 94103 (score: 0.78)
Output: {"best_match": "789 Main St, San Francisco, CA 94102", "confidence": 0.88, "reason": "'mane' is phonetic misspelling of 'Main', 'san fran' = San Francisco"}

Example 4:
Input: "URB villa carolina 123 calle a carolina pr"
Candidates:
1. URB Villa Carolina, 123 Calle A, Carolina, PR 00983 (score: 0.94)
2. 123 Calle A, Carolina, PR 00983 (score: 0.85)
Output: {"best_match": "URB Villa Carolina, 123 Calle A, Carolina, PR 00983", "confidence": 0.96, "reason": "Urbanization matches, all address components match, Puerto Rico address"}

Example 5:
Input: "999 nonexistent blvd nowhere"
Candidates:
1. 909 Nexus Blvd, Newark, NJ 07102 (score: 0.45)
2. 999 North Blvd, Newark, NJ 07103 (score: 0.42)
Output: {"best_match": null, "confidence": 0.0, "reason": "No good matches - candidates don't match input well enough"}
"""


class NovaAddressReranker:
    """Nova-based reranker for address completion."""

    def __init__(
        self,
        region: str = "us-east-1",
        model: str = "us.amazon.nova-lite-v1:0",
        timeout: int = 30
    ):
        """Initialize Nova reranker.

        Args:
            region: AWS region for Bedrock
            model: Nova model ID (lite recommended for speed/cost)
            timeout: API timeout in seconds
        """
        self.model = model
        self.client = None
        self.available = False

        try:
            boto_config = BotoConfig(
                read_timeout=timeout,
                connect_timeout=15,
                retries={"max_attempts": 2},
            )
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=boto_config
            )
            session = boto3.Session()
            if session.get_credentials() is not None:
                self.available = True
                logger.info(f"NovaAddressReranker: Connected ({model})")
            else:
                logger.warning("NovaAddressReranker: No AWS credentials - LITE MODE")
        except NoCredentialsError:
            logger.warning("NovaAddressReranker: AWS credentials not found - LITE MODE")
        except Exception as e:
            logger.error(f"NovaAddressReranker: Init failed - {e}")

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        return_reasoning: bool = False
    ) -> RerankerResult:
        """Rerank candidates and select best match.

        Args:
            query: Original address query
            candidates: List of candidate dicts with full_address and score
            return_reasoning: Include explanation in result

        Returns:
            RerankerResult with best match and confidence
        """
        if not candidates:
            return RerankerResult(match=None, confidence=0.0)

        # If only one high-confidence candidate, skip LLM
        if len(candidates) == 1 and candidates[0].get("score", 0) >= 0.9:
            return RerankerResult(
                match=candidates[0].get("full_address"),
                confidence=candidates[0].get("score", 0.9),
                crid=candidates[0].get("crid")
            )

        # If Nova unavailable, return best vector match
        if not self.available or not self.client:
            best = candidates[0]
            return RerankerResult(
                match=best.get("full_address"),
                confidence=best.get("score", 0.7),
                crid=best.get("crid"),
                reason="Fallback: Nova unavailable, using vector similarity"
            )

        # Format candidates for prompt
        candidates_text = "\n".join([
            f"{i+1}. {c.get('full_address', 'N/A')} (score: {c.get('score', 0):.2f})"
            for i, c in enumerate(candidates[:5])  # Top 5 only to save tokens
        ])

        prompt = f"""{FEW_SHOT_EXAMPLES}

Now complete this task:
Input: "{query}"
Candidates:
{candidates_text}

Select the best matching address. Respond with ONLY valid JSON:
{{"best_match": "full address or null", "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

        try:
            response = self.client.invoke_model(
                modelId=self.model,
                body=json.dumps({
                    "messages": [{"role": "user", "content": [{"text": prompt}]}],
                    "inferenceConfig": {
                        "temperature": 0.1,  # Low for deterministic output
                        "maxTokens": 200
                    }
                })
            )

            result = json.loads(response["body"].read())
            output_text = result.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "{}")

            # Parse JSON response
            try:
                # Handle potential markdown code blocks
                if "```" in output_text:
                    output_text = output_text.split("```")[1]
                    if output_text.startswith("json"):
                        output_text = output_text[4:]

                parsed = json.loads(output_text.strip())

                best_match = parsed.get("best_match")
                confidence = float(parsed.get("confidence", 0.8))
                reason = parsed.get("reason") if return_reasoning else None

                # Find CRID for matched address
                crid = None
                if best_match:
                    for c in candidates:
                        if c.get("full_address") == best_match:
                            crid = c.get("crid")
                            break

                return RerankerResult(
                    match=best_match,
                    confidence=confidence,
                    reason=reason,
                    crid=crid
                )

            except json.JSONDecodeError:
                logger.warning(f"NovaAddressReranker: Failed to parse response: {output_text[:100]}")
                # Fallback to best vector match
                best = candidates[0]
                return RerankerResult(
                    match=best.get("full_address"),
                    confidence=best.get("score", 0.7) * 0.9,  # Slight penalty
                    crid=best.get("crid"),
                    reason="Fallback: JSON parse error"
                )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            logger.error(f"NovaAddressReranker: API error - {error_code}")

            if error_code == "ThrottlingException":
                # On throttle, use vector results
                best = candidates[0]
                return RerankerResult(
                    match=best.get("full_address"),
                    confidence=best.get("score", 0.7),
                    crid=best.get("crid"),
                    reason="Fallback: API throttled"
                )

            return RerankerResult(match=None, confidence=0.0, reason=f"API error: {error_code}")

        except Exception as e:
            logger.error(f"NovaAddressReranker: Rerank error - {e}")
            # Fallback to best vector match
            if candidates:
                best = candidates[0]
                return RerankerResult(
                    match=best.get("full_address"),
                    confidence=best.get("score", 0.7),
                    crid=best.get("crid"),
                    reason=f"Fallback: {str(e)[:50]}"
                )
            return RerankerResult(match=None, confidence=0.0)

    async def complete_partial(
        self,
        partial: str,
        candidates: List[Dict[str, Any]],
        require_components: Optional[List[str]] = None
    ) -> RerankerResult:
        """Complete a partial address using candidates.

        Similar to rerank but focuses on filling in missing components.

        Args:
            partial: Partial address input
            candidates: Vector search candidates
            require_components: Required address components (city, state, zip)

        Returns:
            RerankerResult with completed address
        """
        if not candidates:
            return RerankerResult(match=None, confidence=0.0)

        # Check what components are missing
        partial_lower = partial.lower()
        has_zip = any(word.isdigit() and len(word) == 5 for word in partial.split())
        has_state = any(len(word) == 2 and word.upper() == word and word.isalpha()
                       for word in partial.replace(",", " ").split())

        # If all components present, just rerank
        if has_zip and has_state:
            return await self.rerank(partial, candidates)

        # Otherwise, use completion-focused prompt
        if not self.available or not self.client:
            best = candidates[0]
            return RerankerResult(
                match=best.get("full_address"),
                confidence=best.get("score", 0.7),
                crid=best.get("crid"),
                reason="Fallback: Nova unavailable"
            )

        candidates_text = "\n".join([
            f"{i+1}. {c.get('full_address', 'N/A')} (score: {c.get('score', 0):.2f})"
            for i, c in enumerate(candidates[:5])
        ])

        missing = []
        if not has_zip:
            missing.append("ZIP code")
        if not has_state:
            missing.append("state")

        prompt = f"""Complete this partial address by matching to the best candidate.

Partial address: "{partial}"
Missing: {', '.join(missing)}

Candidates from database:
{candidates_text}

Select the best match and return the COMPLETE address. Respond with ONLY valid JSON:
{{"best_match": "complete full address", "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

        try:
            response = self.client.invoke_model(
                modelId=self.model,
                body=json.dumps({
                    "messages": [{"role": "user", "content": [{"text": prompt}]}],
                    "inferenceConfig": {
                        "temperature": 0.1,
                        "maxTokens": 200
                    }
                })
            )

            result = json.loads(response["body"].read())
            output_text = result.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "{}")

            try:
                if "```" in output_text:
                    output_text = output_text.split("```")[1]
                    if output_text.startswith("json"):
                        output_text = output_text[4:]

                parsed = json.loads(output_text.strip())

                crid = None
                best_match = parsed.get("best_match")
                if best_match:
                    for c in candidates:
                        if c.get("full_address") == best_match:
                            crid = c.get("crid")
                            break

                return RerankerResult(
                    match=best_match,
                    confidence=float(parsed.get("confidence", 0.8)),
                    reason=parsed.get("reason"),
                    crid=crid
                )

            except json.JSONDecodeError:
                best = candidates[0]
                return RerankerResult(
                    match=best.get("full_address"),
                    confidence=best.get("score", 0.7) * 0.9,
                    crid=best.get("crid"),
                    reason="Fallback: JSON parse error"
                )

        except Exception as e:
            logger.error(f"NovaAddressReranker: Complete error - {e}")
            if candidates:
                best = candidates[0]
                return RerankerResult(
                    match=best.get("full_address"),
                    confidence=best.get("score", 0.7),
                    crid=best.get("crid"),
                    reason=f"Fallback: {str(e)[:50]}"
                )
            return RerankerResult(match=None, confidence=0.0)
