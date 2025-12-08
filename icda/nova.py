import boto3
from botocore.exceptions import ClientError

from .database import CustomerDB


class NovaClient:
    __slots__ = ("client", "model", "available", "db")

    _PROMPT = "You are ICDA, an AI assistant for customer data queries. Be concise. Never provide SSN, financial, or health info. You have access to conversation history - use it to maintain context."

    TOOLS = [
        {"toolSpec": {"name": "lookup_crid", "description": "Look up customer by CRID",
            "inputSchema": {"json": {"type": "object", "properties": {"crid": {"type": "string"}}, "required": ["crid"]}}}},
        {"toolSpec": {"name": "search_customers", "description": "Search customers by state, city, or move count",
            "inputSchema": {"json": {"type": "object", "properties": {
                "state": {"type": "string"}, "city": {"type": "string"},
                "min_move_count": {"type": "integer"}, "limit": {"type": "integer"}}}}}},
        {"toolSpec": {"name": "get_stats", "description": "Get customer statistics",
            "inputSchema": {"json": {"type": "object", "properties": {}}}}}
    ]

    def __init__(self, region: str, model: str, db: CustomerDB):
        self.model = model
        self.db = db
        try:
            self.client = boto3.client("bedrock-runtime", region_name=region)
            self.available = True
        except Exception as e:
            print(f"Nova init failed: {e}")
            self.available = False

    def _converse(self, messages: list) -> dict:
        return self.client.converse(
            modelId=self.model,
            messages=messages,
            system=[{"text": self._PROMPT}],
            toolConfig={"tools": self.TOOLS, "toolChoice": {"auto": {}}},
            inferenceConfig={"maxTokens": 4096, "temperature": 0.1}
        )

    async def query(self, text: str, history: list[dict] | None = None) -> dict:
        """
        Query Nova with optional conversation history.

        Args:
            text: The user's query
            history: Previous messages in Bedrock format [{"role": "user/assistant", "content": [{"text": "..."}]}]
                     IMPORTANT: History must only contain text content, not toolUse blocks!

        Returns:
            dict with success, response, and optional tool used
        """
        if not self.available:
            return {"success": False, "error": "Nova not available"}

        try:
            # Build messages: history + current query
            # Filter history to ensure only text content (no toolUse blocks that would require toolResult)
            messages = []
            if history:
                for msg in history:
                    # Only include messages with pure text content
                    clean_content = [b for b in msg.get("content", []) if "text" in b]
                    if clean_content:
                        messages.append({"role": msg["role"], "content": clean_content})
            messages.append({"role": "user", "content": [{"text": text}]})

            resp = self._converse(messages)
            content = resp["output"]["message"]["content"]

            if tool := next((b["toolUse"] for b in content if "toolUse" in b), None):
                result = self._execute_tool(tool["name"], tool["input"])

                # Continue with tool result (this is a self-contained exchange, not stored in history)
                follow_messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": [{"toolResult": {"toolUseId": tool["toolUseId"], "content": [{"json": result}]}}]}
                ]
                follow = self._converse(follow_messages)

                if out := next((b["text"] for b in follow["output"]["message"]["content"] if "text" in b), None):
                    return {"success": True, "response": out, "tool": tool["name"]}

            if out := next((b["text"] for b in content if "text" in b), None):
                return {"success": True, "response": out}
            return {"success": False, "error": "No response"}

        except ClientError as e:
            return {"success": False, "error": f"Nova: {e.response['Error']['Message']}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_tool(self, name: str, params: dict) -> dict:
        match name:
            case "lookup_crid":
                return self.db.lookup(params.get("crid", ""))
            case "search_customers":
                return self.db.search(
                    state=params.get("state"), city=params.get("city"), name=params.get("name"),
                    min_moves=params.get("min_move_count"), limit=params.get("limit", 10)
                )
            case "get_stats":
                return self.db.stats()
        return {"success": False, "error": f"Unknown tool: {name}"}