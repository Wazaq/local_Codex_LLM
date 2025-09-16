import os
import os.path as _p

# Backward-compatible environment mapping
MCP_URL = os.getenv(
    'MCP_URL',
    os.getenv('CODEX_MCP_URL', 'https://neural-nexus-palace.wazaqglim.workers.dev/mcp')
)

MODEL_NAME = os.getenv('CODEX_MODEL', os.getenv('MODEL_NAME', 'qwen2.5-coder:32b'))
NUM_CTX = int(os.getenv('CODEX_CONTEXT', os.getenv('NUM_CTX', '8192')))
MCP_SEARCH_TOOL = os.getenv('CODEX_MCP_TOOL_SEARCH', 'brain_ai_library_search')
LOG_LEVEL = os.getenv('CODEX_LOG_LEVEL', os.getenv('LOG_LEVEL', 'INFO')).upper()
MCP_RETRY_ATTEMPTS = int(os.getenv('CODEX_MCP_RETRY_ATTEMPTS', os.getenv('MCP_RETRY_ATTEMPTS', '3')))
MCP_TIMEOUT_SECONDS = float(os.getenv('CODEX_MCP_TIMEOUT_SECONDS', os.getenv('MCP_TIMEOUT_SECONDS', '10')))
MCP_BACKOFF_BASE = float(os.getenv('CODEX_MCP_BACKOFF_BASE', os.getenv('MCP_BACKOFF_BASE', '1.0')))
HTTP_FALLBACK_TIMEOUT = float(os.getenv('CODEX_HTTP_FALLBACK_TIMEOUT', os.getenv('HTTP_FALLBACK_TIMEOUT', '5')))
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv('CODEX_CIRCUIT_BREAKER_THRESHOLD', os.getenv('CIRCUIT_BREAKER_THRESHOLD', '5')))
CIRCUIT_BREAKER_COOLDOWN = float(os.getenv('CODEX_CIRCUIT_BREAKER_COOLDOWN_SECONDS', os.getenv('CIRCUIT_BREAKER_COOLDOWN_SECONDS', '30')))
ENABLE_CONTENT_FILTER = os.getenv('CODEX_ENABLE_CONTENT_FILTER', os.getenv('ENABLE_CONTENT_FILTER', 'false')).lower() == 'true'
MAX_REQUESTS_PER_HOUR = int(os.getenv('CODEX_MAX_REQUESTS_PER_HOUR', os.getenv('MAX_REQUESTS_PER_HOUR', '1000')))
AIL_RPM_LIMIT = int(os.getenv('CODEX_AIL_REQUESTS_PER_MIN', os.getenv('AIL_RPM_LIMIT', '5')))
AIL_PROXY_TIMEOUT = float(os.getenv('CODEX_AIL_PROXY_TIMEOUT_SECONDS', os.getenv('AIL_PROXY_TIMEOUT', '180')))
CHAT_RPM_LIMIT = int(os.getenv('CODEX_CHAT_REQUESTS_PER_MIN', os.getenv('CHAT_RPM_LIMIT', '5')))
BLOCKED_KEYWORDS_FILE = os.getenv('CODEX_BLOCKED_KEYWORDS_FILE', os.getenv('BLOCKED_KEYWORDS_FILE', ''))
ALLOW_LIST_DOMAINS = [
    d.strip().lower() for d in os.getenv(
        'CODEX_ALLOW_LIST_DOMAINS', os.getenv('ALLOW_LIST_DOMAINS', 'neural-nexus-palace.wazaqglim.workers.dev')
    ).split(',') if d.strip()
]

_default_logs_dir = _p.join(_p.dirname(__file__), 'logs')
SECURITY_LOG_FILE = os.getenv(
    'CODEX_SECURITY_LOG_FILE', os.getenv('SECURITY_LOG_FILE', _p.normpath(_p.join(_default_logs_dir, 'security.log')))
)

SHUTDOWN_TIMEOUT = float(os.getenv('CODEX_SHUTDOWN_TIMEOUT', os.getenv('SHUTDOWN_TIMEOUT', '30')))
ENABLE_SHUTDOWN_ENDPOINT = os.getenv('CODEX_ENABLE_SHUTDOWN_ENDPOINT', os.getenv('ENABLE_SHUTDOWN_ENDPOINT', 'false')).lower() == 'true'
SESSION_TOKEN_LIMIT = int(os.getenv('CODEX_SESSION_TOKEN_LIMIT', os.getenv('SESSION_TOKEN_LIMIT', '100000')))
SESSION_TTL_SECONDS = int(os.getenv('CODEX_SESSION_TTL_SECONDS', os.getenv('SESSION_TTL_SECONDS', '0')))
SESSION_BACKEND = os.getenv('CODEX_SESSION_BACKEND', os.getenv('SESSION_BACKEND', 'memory')).lower()
REDIS_URL = os.getenv('CODEX_REDIS_URL', os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
SESSION_STORAGE_PATH = os.getenv('CODEX_SESSION_STORAGE_PATH', os.getenv('SESSION_STORAGE_PATH', 'data/sessions.json'))
SESSION_TTL_DAYS = int(os.getenv('CODEX_SESSION_TTL_DAYS', os.getenv('SESSION_TTL_DAYS', '0')))
BRIDGE_AI_ID = os.getenv('CODEX_BRIDGE_AI_ID', os.getenv('BRIDGE_AI_ID', 'codex'))

# Ollama URL is fixed in monolith to localhost
OLLAMA_BASE_URL = os.getenv('OLLAMA_URL', os.getenv('CODEX_OLLAMA_URL', 'http://localhost:11434'))
