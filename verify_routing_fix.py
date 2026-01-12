"""Quick verification script for the routing fix."""
import asyncio
import sys
sys.path.insert(0, ".")

from icda.agents.intent_agent import IntentAgent
from icda.agents.model_router import ModelRouter
from icda.classifier import QueryComplexity
from icda.agents.models import ModelTier

async def test_routing():
    intent_agent = IntentAgent()
    model_router = ModelRouter()
    
    print("=" * 60)
    print("ICDA ROUTING FIX VERIFICATION")
    print("=" * 60)
    
    # The critical test case
    query = "show me all the Texas customers who are inactive and have moved from California"
    print(f"\nQuery: {query}")
    
    intent = await intent_agent.classify(query)
    routing = model_router.route(intent)
    
    print(f"\nResults:")
    print(f"  Complexity: {intent.complexity.value}")
    print(f"  Reasons: {intent.raw_signals.get('complexity_reasons', [])}")
    print(f"  Model Tier: {routing.model_tier.value}")
    print(f"  Routing Reason: {routing.reason}")
    print(f"  Suggested Tools: {intent.suggested_tools[:5]}")
    
    # Assertions
    passed = True
    if intent.complexity != QueryComplexity.COMPLEX:
        print(f"\n❌ FAIL: Expected COMPLEX, got {intent.complexity.value}")
        passed = False
    else:
        print(f"\n✅ PASS: Complexity is COMPLEX")
    
    if routing.model_tier != ModelTier.PRO:
        print(f"❌ FAIL: Expected PRO, got {routing.model_tier.value}")
        passed = False
    else:
        print(f"✅ PASS: Routes to Nova Pro")
    
    if "customers_moved_from" not in intent.suggested_tools:
        print(f"❌ FAIL: customers_moved_from not in suggested tools")
        passed = False
    else:
        print(f"✅ PASS: customers_moved_from tool suggested")
    
    print("\n" + "=" * 60)
    if passed:
        print("ALL TESTS PASSED! ✅")
    else:
        print("SOME TESTS FAILED! ❌")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_routing())
