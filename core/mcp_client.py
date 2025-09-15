import time
import random
import requests

from . import utils
import config


class HTTPMCPClient:
    def __init__(self, mcp_url: str):
        self.mcp_url = mcp_url

    def call_mcp_function(self, tool_name, arguments, timeout=None):
        payload = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments}
        }

        if not utils.cb_can_call():
            return {"error": "MCP circuit open"}

        attempts = max(1, config.MCP_RETRY_ATTEMPTS)
        for attempt in range(attempts):
            try:
                response = requests.post(self.mcp_url, json=payload, timeout=(timeout or config.MCP_TIMEOUT_SECONDS))
                response.raise_for_status()
                utils.cb_record_success()
                return response.json()
            except requests.exceptions.RequestException as e:
                utils.cb_record_failure()
                if attempt >= attempts - 1:
                    return {"error": f"MCP call failed: {str(e)}"}
                backoff = config.MCP_BACKOFF_BASE * (2 ** attempt)
                jitter = backoff * random.uniform(0, 0.2)
                time.sleep(backoff + jitter)

    def list_tools(self):
        payload = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tools/list",
            "params": {}
        }
        try:
            response = requests.post(self.mcp_url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"MCP list failed: {str(e)}"}

