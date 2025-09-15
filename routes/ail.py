from flask import Blueprint, request, jsonify, current_app
import json

import config
from core.security import contains_blocked
from core.utils import extract_results_from_mcp, to_text_list, http_ail_search


ail_bp = Blueprint('ail', __name__)


def _extract_ai_id():
    data = request.get_json(silent=True) or {}
    return (request.headers.get('X-AI-Id') or data.get('ai_id') or '').strip()


@ail_bp.route('/ail/search/semantic', methods=['POST'])
def ail_semantic_search():
    metrics = current_app.config['metrics']
    ail_client = current_app.config['ail_client']
    ai_id = _extract_ai_id()
    # AI per-minute rate limit
    if not current_app.config['ai_rate_limit_check'](ai_id, '/ail/search/semantic', config.AIL_RPM_LIMIT):
        metrics.inc_counter('codex_rate_limit_block_total', {"endpoint": "/ail/search/semantic", "ai_id": ai_id or "unknown"})
        return jsonify({"error": "Rate limit exceeded for AI.", "ai_id": ai_id}), 429

    data = request.get_json(silent=True) or {}
    query = data.get('query') or data.get('q')
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400
    args = {
        "query": query,
        **({"top_k": data.get('top_k')} if data.get('top_k') is not None else {}),
        **({"min_relevance": data.get('min_relevance')} if data.get('min_relevance') is not None else {}),
        **({"domains": data.get('domains')} if data.get('domains') is not None else {}),
        **({"include_content": data.get('include_content')} if data.get('include_content') is not None else {}),
    }
    if config.ENABLE_CONTENT_FILTER:
        term = contains_blocked(query)
        if term:
            return jsonify({"error": "Query blocked by content policy."}), 400

    import time as _t
    t0 = _t.time()
    metrics.inc_counter('codex_requests_total', {"endpoint": "/ail/search/semantic", "method": "POST"})
    resp = ail_client.call_mcp_function('ai_lib_semantic_search', args, timeout=config.AIL_PROXY_TIMEOUT)
    dt = _t.time() - t0
    metrics.observe_histogram('codex_request_duration_seconds', dt, {"endpoint": "/ail/search/semantic"})
    if isinstance(resp, dict) and resp.get('error'):
        metrics.inc_counter('codex_request_errors_total', {"endpoint": "/ail/search/semantic"})
        return jsonify({"error": resp.get('error')}), 502

    raw = resp.get('result') if isinstance(resp, dict) else resp
    if isinstance(raw, dict) and 'text' in raw:
        raw = raw['text']
    if isinstance(raw, dict) and 'content' in raw:
        raw = raw['content']
    if isinstance(raw, dict):
        try:
            raw = json.dumps(raw)
        except Exception:
            raw = str(raw)
    normalized = _normalize_text_response('semantic_search', raw if isinstance(raw, str) else str(raw))
    return jsonify({"ok": True, "ai_id": ai_id, "data": normalized})


def _normalize_text_response(kind: str, raw: str):
    import re as _re
    _re_matches = _re.compile(r"(\d+)\s+matches", _re.IGNORECASE)
    _re_numbered = _re.compile(r"^\s*\d+\.\s+(.*?)(?:\s*\((\d+)%\))?\s*$")
    _re_records = _re.compile(r"(\d+)\s+records", _re.IGNORECASE)
    raw = raw or ""
    out = {"ok": True, "kind": kind, "raw": raw}
    try:
        if kind == 'semantic_search':
            m = _re_matches.search(raw)
            if m:
                out["count"] = int(m.group(1))
            items = []
            for line in (raw.splitlines() if isinstance(raw, str) else []):
                m2 = _re_numbered.match(line)
                if m2:
                    items.append({"title": m2.group(1), "relevance_pct": int(m2.group(2)) if m2.group(2) else None})
            if items:
                out["items"] = items
        elif kind in ('memory_read', 'memory_create', 'memory_update'):
            m = _re_records.search(raw)
            if m:
                out["count"] = int(m.group(1))
    except Exception:
        pass
    return out


@ail_bp.route('/ail/memory/read', methods=['POST'])
def ail_memory_read():
    metrics = current_app.config['metrics']
    ail_client = current_app.config['ail_client']
    ai_id = _extract_ai_id()
    if not current_app.config['ai_rate_limit_check'](ai_id, '/ail/memory/read', config.AIL_RPM_LIMIT):
        metrics.inc_counter('codex_rate_limit_block_total', {"endpoint": "/ail/memory/read", "ai_id": ai_id or "unknown"})
        return jsonify({"error": "Rate limit exceeded for AI.", "ai_id": ai_id}), 429

    data = request.get_json(silent=True) or {}
    args = {
        **({"table": data.get('table')} if data.get('table') is not None else {}),
        **({"filters": data.get('filters')} if data.get('filters') is not None else {}),
        **({"limit": data.get('limit')} if data.get('limit') is not None else {}),
    }
    import time as _t
    t0 = _t.time()
    metrics.inc_counter('codex_requests_total', {"endpoint": "/ail/memory/read", "method": "POST"})
    resp = ail_client.call_mcp_function('neural_memory_manager_read', args, timeout=config.AIL_PROXY_TIMEOUT)
    dt = _t.time() - t0
    metrics.observe_histogram('codex_request_duration_seconds', dt, {"endpoint": "/ail/memory/read"})
    if isinstance(resp, dict) and resp.get('error'):
        metrics.inc_counter('codex_request_errors_total', {"endpoint": "/ail/memory/read"})
        return jsonify({"error": resp.get('error')}), 502
    raw = resp.get('result') if isinstance(resp, dict) else resp
    if isinstance(raw, dict):
        raw = raw.get('text') or raw.get('content') or json.dumps(raw)
    normalized = _normalize_text_response('memory_read', raw if isinstance(raw, str) else str(raw))
    return jsonify({"ok": True, "ai_id": ai_id, "data": normalized})


@ail_bp.route('/ail/memory/create', methods=['POST'])
def ail_memory_create():
    metrics = current_app.config['metrics']
    ail_client = current_app.config['ail_client']
    ai_id = _extract_ai_id()
    if not current_app.config['ai_rate_limit_check'](ai_id, '/ail/memory/create', config.AIL_RPM_LIMIT):
        metrics.inc_counter('codex_rate_limit_block_total', {"endpoint": "/ail/memory/create", "ai_id": ai_id or "unknown"})
        return jsonify({"error": "Rate limit exceeded for AI.", "ai_id": ai_id}), 429

    data = request.get_json(silent=True) or {}
    args = {k: v for k, v in (data or {}).items() if k != 'ai_id'}
    import time as _t
    t0 = _t.time()
    metrics.inc_counter('codex_requests_total', {"endpoint": "/ail/memory/create", "method": "POST"})
    resp = ail_client.call_mcp_function('neural_memory_manager_create', args, timeout=config.AIL_PROXY_TIMEOUT)
    dt = _t.time() - t0
    metrics.observe_histogram('codex_request_duration_seconds', dt, {"endpoint": "/ail/memory/create"})
    if isinstance(resp, dict) and resp.get('error'):
        metrics.inc_counter('codex_request_errors_total', {"endpoint": "/ail/memory/create"})
        return jsonify({"error": resp.get('error')}), 502
    raw = resp.get('result') if isinstance(resp, dict) else resp
    if isinstance(raw, dict):
        raw = raw.get('text') or raw.get('content') or json.dumps(raw)
    normalized = _normalize_text_response('memory_create', raw if isinstance(raw, str) else str(raw))
    return jsonify({"ok": True, "ai_id": ai_id, "data": normalized})


@ail_bp.route('/ail/memory/update', methods=['PUT'])
def ail_memory_update():
    metrics = current_app.config['metrics']
    ail_client = current_app.config['ail_client']
    ai_id = _extract_ai_id()
    if not current_app.config['ai_rate_limit_check'](ai_id, '/ail/memory/update', config.AIL_RPM_LIMIT):
        metrics.inc_counter('codex_rate_limit_block_total', {"endpoint": "/ail/memory/update", "ai_id": ai_id or "unknown"})
        return jsonify({"error": "Rate limit exceeded for AI.", "ai_id": ai_id}), 429

    data = request.get_json(silent=True) or {}
    args = {k: v for k, v in (data or {}).items() if k != 'ai_id'}
    import time as _t
    t0 = _t.time()
    metrics.inc_counter('codex_requests_total', {"endpoint": "/ail/memory/update", "method": "PUT"})
    resp = ail_client.call_mcp_function('neural_memory_manager_update', args, timeout=config.AIL_PROXY_TIMEOUT)
    dt = _t.time() - t0
    metrics.observe_histogram('codex_request_duration_seconds', dt, {"endpoint": "/ail/memory/update"})
    if isinstance(resp, dict) and resp.get('error'):
        metrics.inc_counter('codex_request_errors_total', {"endpoint": "/ail/memory/update"})
        return jsonify({"error": resp.get('error')}), 502
    raw = resp.get('result') if isinstance(resp, dict) else resp
    if isinstance(raw, dict):
        raw = raw.get('text') or raw.get('content') or json.dumps(raw)
    normalized = _normalize_text_response('memory_update', raw if isinstance(raw, str) else str(raw))
    return jsonify({"ok": True, "ai_id": ai_id, "data": normalized})


@ail_bp.route('/ail/context', methods=['GET'])
def ail_get_context():
    metrics = current_app.config['metrics']
    ail_client = current_app.config['ail_client']
    ai_id = (request.headers.get('X-AI-Id') or '').strip()
    if not current_app.config['ai_rate_limit_check'](ai_id, '/ail/context', config.AIL_RPM_LIMIT):
        metrics.inc_counter('codex_rate_limit_block_total', {"endpoint": "/ail/context", "ai_id": ai_id or "unknown"})
        return jsonify({"error": "Rate limit exceeded for AI.", "ai_id": ai_id}), 429

    include_domain_stats = request.args.get('include_domain_stats', 'true').lower() != 'false'
    args = {"include_domain_stats": include_domain_stats}
    import time as _t
    t0 = _t.time()
    metrics.inc_counter('codex_requests_total', {"endpoint": "/ail/context", "method": "GET"})
    resp = ail_client.call_mcp_function('ai_lib_get_context', args, timeout=config.AIL_PROXY_TIMEOUT)
    dt = _t.time() - t0
    metrics.observe_histogram('codex_request_duration_seconds', dt, {"endpoint": "/ail/context"})
    if isinstance(resp, dict) and resp.get('error'):
        metrics.inc_counter('codex_request_errors_total', {"endpoint": "/ail/context"})
        return jsonify({"error": resp.get('error')}), 502
    raw = resp.get('result') if isinstance(resp, dict) else resp
    if isinstance(raw, dict):
        raw = raw.get('text') or raw.get('content') or json.dumps(raw)
    normalized = _normalize_text_response('context', raw if isinstance(raw, str) else str(raw))
    return jsonify({"ok": True, "ai_id": ai_id, "data": normalized})


@ail_bp.route('/mcp', methods=['POST'])
def mcp_http_server():
    # Minimal MCP interface passthrough for a few tools using the internal routes
    app = current_app
    req = request.get_json(silent=True) or {}
    method = req.get('method')
    req_id = req.get('id')
    if method == 'tools/list':
        tools = [
            {
                "name": "codex_chat",
                "description": "Chat with Codex AI using her experimental personality",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "temperature": {"type": "number", "default": 0.7},
                        "top_p": {"type": "number", "default": 0.9},
                        "seed": {"type": ["integer", "null"], "default": None},
                        "debug": {"type": "boolean", "default": False}
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "codex_chat_with_context",
                "description": "Chat with Codex using custom personality/memory context overrides",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "personality": {"type": ["array", "string"], "default": []},
                        "memories": {"type": ["array", "string"], "default": []},
                        "temperature": {"type": "number", "default": 0.7},
                        "top_p": {"type": "number", "default": 0.9},
                        "seed": {"type": ["integer", "null"], "default": None},
                        "debug": {"type": "boolean", "default": False}
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "codex_search_ail",
                "description": "Search AI Library via Codex bridge",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "domain": {"type": ["string", "null"], "default": None},
                        "limit": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                }
            }
        ]
        return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}})
    elif method == 'tools/call':
        params = req.get('params', {})
        name = params.get('name')
        args = params.get('arguments', {})
        if name == 'codex_chat':
            payload = {
                "message": args.get('message', ''),
                "temperature": args.get('temperature', 0.7),
                "top_p": args.get('top_p', 0.9),
                "seed": args.get('seed', None),
                "debug": args.get('debug', False)
            }
            with app.test_request_context(json=payload):
                # import lazily to avoid cycles
                from .chat import chat_with_codex
                res = chat_with_codex()
                resp_obj = res[0] if isinstance(res, tuple) else res
                return jsonify({"jsonrpc": "2.0", "id": req_id, "result": json.loads(resp_obj.get_data(as_text=True))})
        elif name == 'codex_chat_with_context':
            message = args.get('message', '')
            personality = args.get('personality', [])
            memories = args.get('memories', [])
            if isinstance(personality, str):
                personality = [personality]
            if isinstance(memories, str):
                memories = [memories]
            personality_lines = to_text_list(personality)
            memory_lines = to_text_list(memories)
            system_prompt = (
                "You are Codex, a helpful AI with a distinct personality.\n"
                "Use the provided personality traits and recent memories as context when responding.\n"
                "If memories are not relevant, prioritize the user's message. Be concise and concrete.\n\n"
                f"PERSONALITY:\n- " + "\n- ".join(personality_lines) +
                ("\n\nRECENT MEMORIES:\n- " + "\n- ".join(memory_lines) if memory_lines else "")
            )
            chat_payload = {
                "model": config.MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                "stream": False,
                "options": {
                    "num_ctx": config.NUM_CTX,
                    "temperature": args.get('temperature', 0.7),
                    "top_p": args.get('top_p', 0.9),
                    **({"seed": args.get('seed')} if args.get('seed') is not None else {})
                }
            }
            import requests
            r = requests.post(f"{config.OLLAMA_BASE_URL}/api/chat", json=chat_payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            content = data.get('message', {}).get('content') or data.get('response', '')
            return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"response": content}})
        elif name == 'codex_search_ail':
            q = args.get('query', '')
            domain = args.get('domain')
            limit = args.get('limit', 10)
            result = http_ail_search(q, domain=domain, limit=limit)
            return jsonify({"jsonrpc": "2.0", "id": req_id, "result": result})
        else:
            return jsonify({"jsonrpc": "2.0", "id": req_id, "error": {"message": f"Unknown tool: {name}"}})
    else:
        return jsonify({"jsonrpc": "2.0", "id": req_id, "error": {"message": "Unsupported method"}})


@ail_bp.route('/get-codex-personality', methods=['POST'])
def get_codex_personality():
    url = "https://neural-nexus-palace.wazaqglim.workers.dev/mobile/brain-ai-library-search"
    params = {"q": "experimental", "domain": "Codex Personality Room", "limit": 10}
    import requests
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"HTTP call failed: {str(e)}"})


@ail_bp.route('/get-memories', methods=['POST'])
def get_memories():
    url = "https://neural-nexus-palace.wazaqglim.workers.dev/mobile/brain-ai-library-search"
    params = {"q": "Claude", "limit": 10}
    import requests
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"HTTP call failed: {str(e)}"})


@ail_bp.route('/ail-search', methods=['POST'])
def ail_search():
    metrics = current_app.config['metrics']
    data = request.get_json(silent=True) or {}
    search_query = data.get('query', '')
    domain_filter = data.get('domain', None)
    limit = data.get('limit', 10)
    if not search_query:
        return jsonify({"error": "Query parameter required"})
    metrics.inc_counter('codex_requests_total', {"endpoint": "/ail-search", "method": "POST"})
    import requests
    try:
        url = "https://neural-nexus-palace.wazaqglim.workers.dev/mobile/brain-ai-library-search"
        params = {"q": search_query, "limit": limit}
        if domain_filter:
            params["domain"] = domain_filter
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        result = response.json()
        return jsonify({
            "query": search_query,
            "domain": domain_filter or "all",
            "results": result.get('results', []),
            "total_found": result.get('total_results', 0),
            "search_tip": "Use literal terms that exist in content. Try: 'Claude', 'experimental', 'development', 'project'"
        })
    except requests.exceptions.RequestException as e:
        metrics.inc_counter('codex_request_errors_total', {"endpoint": "/ail-search"})
        return jsonify({"error": f"AIL search failed: {str(e)}"})

