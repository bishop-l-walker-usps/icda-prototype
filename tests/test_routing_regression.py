"""Regression tests for ICDA model routing.

Ensures complex queries route to Nova Pro, simple to Micro.
Run with: pytest tests/test_routing_regression.py -v
"""

import pytest
from icda.agents.intent_agent import IntentAgent
from icda.agents.model_router import ModelRouter
from icda.classifier import QueryComplexity
from icda.agents.models import ModelTier


class TestRoutingRegression:
    """Regression tests to prevent routing breakage."""

    @pytest.fixture
    def intent_agent(self):
        return IntentAgent()

    @pytest.fixture
    def model_router(self):
        return ModelRouter()

    # =========================================================================
    # COMPLEXITY CLASSIFICATION TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_multi_condition_is_complex(self, intent_agent):
        """Query with state + status + move_from should be COMPLEX."""
        query = "show me all the Texas customers who are inactive and have moved from California"
        intent = await intent_agent.classify(query)
        
        assert intent.complexity == QueryComplexity.COMPLEX, \
            f"Expected COMPLEX, got {intent.complexity.value}. Reasons: {intent.raw_signals.get('complexity_reasons')}"

    @pytest.mark.asyncio
    async def test_two_conditions_is_complex(self, intent_agent):
        """Query with 2 conditions should be COMPLEX."""
        queries = [
            "inactive customers in Texas",  # status + state
            "customers who moved from California to Nevada",  # move_from + state
            "active customers who moved from Florida",  # status + move_from
        ]
        for query in queries:
            intent = await intent_agent.classify(query)
            assert intent.complexity == QueryComplexity.COMPLEX, \
                f"Query '{query}' should be COMPLEX, got {intent.complexity.value}"

    @pytest.mark.asyncio
    async def test_single_condition_is_medium(self, intent_agent):
        """Query with single filter should be MEDIUM."""
        queries = [
            "inactive customers",
            "customers in Texas", 
            "high movers",
        ]
        for query in queries:
            intent = await intent_agent.classify(query)
            assert intent.complexity in (QueryComplexity.MEDIUM, QueryComplexity.COMPLEX), \
                f"Query '{query}' should be at least MEDIUM, got {intent.complexity.value}"

    @pytest.mark.asyncio
    async def test_simple_lookup_is_simple(self, intent_agent):
        """Simple CRID lookup should be SIMPLE."""
        intent = await intent_agent.classify("look up CRID-001")
        assert intent.complexity == QueryComplexity.SIMPLE

    # =========================================================================
    # TOOL SUGGESTION TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_moved_from_suggests_right_tool(self, intent_agent):
        """'moved from' queries should suggest customers_moved_from tool."""
        query = "show me Texas customers who moved from California"
        intent = await intent_agent.classify(query)
        
        assert "customers_moved_from" in intent.suggested_tools, \
            f"Expected customers_moved_from in tools, got: {intent.suggested_tools}"

    @pytest.mark.asyncio
    async def test_status_suggests_right_tool(self, intent_agent):
        """Status queries should suggest filter_by_status tool."""
        query = "show me inactive customers"
        intent = await intent_agent.classify(query)
        
        assert "filter_by_status" in intent.suggested_tools or \
               "get_inactive_customers" in intent.suggested_tools, \
            f"Expected status tool in suggestions, got: {intent.suggested_tools}"

    # =========================================================================
    # MODEL ROUTING TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_complex_routes_to_pro(self, intent_agent, model_router):
        """COMPLEX queries should route to Nova Pro."""
        query = "Texas customers who are inactive and moved from California"
        intent = await intent_agent.classify(query)
        routing = model_router.route(intent)
        
        assert routing.model_tier == ModelTier.PRO, \
            f"Expected PRO, got {routing.model_tier.value}. Reason: {routing.reason}"

    @pytest.mark.asyncio
    async def test_simple_routes_to_micro(self, intent_agent, model_router):
        """SIMPLE queries should route to Nova Micro."""
        query = "look up CRID-001"
        intent = await intent_agent.classify(query)
        routing = model_router.route(intent)
        
        assert routing.model_tier == ModelTier.MICRO, \
            f"Expected MICRO, got {routing.model_tier.value}. Reason: {routing.reason}"

    # =========================================================================
    # COMPREHENSIVE TEST MATRIX
    # =========================================================================

    ROUTING_CASES = [
        # (query, expected_complexity, expected_tier, test_name)
        ("look up CRID-001", QueryComplexity.SIMPLE, ModelTier.MICRO, "simple_lookup"),
        ("how many customers?", QueryComplexity.SIMPLE, ModelTier.MICRO, "simple_count"),
        ("customers in Nevada", QueryComplexity.MEDIUM, ModelTier.LITE, "single_state"),
        ("inactive customers", QueryComplexity.MEDIUM, ModelTier.LITE, "single_status"),
        ("inactive customers in Texas", QueryComplexity.COMPLEX, ModelTier.PRO, "status_state"),
        ("Texas customers moved from California", QueryComplexity.COMPLEX, ModelTier.PRO, "state_move"),
        ("analyze migration patterns", QueryComplexity.COMPLEX, ModelTier.PRO, "analysis"),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,expected_complexity,expected_tier,name", ROUTING_CASES)
    async def test_routing_matrix(self, intent_agent, model_router, query, expected_complexity, expected_tier, name):
        """Test routing decision matrix."""
        intent = await intent_agent.classify(query)
        routing = model_router.route(intent)
        
        # Check complexity
        assert intent.complexity == expected_complexity, \
            f"[{name}] Complexity: got {intent.complexity.value}, expected {expected_complexity.value}"
        
        # Check routing
        assert routing.model_tier == expected_tier, \
            f"[{name}] Routing: got {routing.model_tier.value}, expected {expected_tier.value}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
