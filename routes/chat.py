from flask import Blueprint, request, jsonify, Response, current_app
import hashlib
import json
import time

import config
from core.utils import extract_results_from_mcp, http_ail_search, now_iso
from core.session_store import approx_tokens, compact_session_if_needed


chat_bp = Blueprint('chat', __name__)


@chat_bp.route('/chat-with-codex', methods=['POST'])
def chat_with_codex():
    app = current_app
    metrics = app.config['metrics']
    sessions = app.config['sessions']
    ail_client = app.config['ail_client']
    security_logger = app.config['security_logger']

    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '')
    include_debug = bool(data.get('debug', False))
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    seed = data.get('seed', None)
    session_id = data.get('session_id')
    requester_ai_id = (data.get('ai_id') or request.headers.get('X-AI-Id') or 'codex').strip()

    # Per-AI rate limiting for chat
    if not app.config['ai_rate_limit_check'](requester_ai_id, '/chat-with-codex', config.CHAT_RPM_LIMIT):
        metrics.inc_counter('codex_rate_limit_block_total', {"endpoint": "/chat-with-codex", "ai_id": requester_ai_id or "unknown"})
        return jsonify({"error": "Rate limit exceeded for AI.", "ai_id": requester_ai_id}), 429

    t_start = time.time()
    metrics.inc_counter('codex_requests_total', {"endpoint": "/chat-with-codex", "method": "POST"})

    # Per-IP rate limiting
    ip = request.headers.get('X-Forwarded-For', request.remote_addr or 'unknown').split(',')[0].strip()
    if not app.config['rate_limit_check'](ip, config.MAX_REQUESTS_PER_HOUR):
        try:
            security_logger.info(json.dumps({
                "timestamp": now_iso(),
                "event": "rate_limit_block",
                "ip": ip
            }))
        except Exception:
            pass
        metrics.inc_counter('codex_rate_limit_block_total', {"endpoint": "/chat-with-codex"})
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

    # Content filter for inbound message
    if config.ENABLE_CONTENT_FILTER:
        term = app.config['contains_blocked'](user_message)
        if term:
            try:
                security_logger.info(json.dumps({
                    "timestamp": now_iso(),
                    "event": "content_blocked",
                    "ip": ip,
                    "reason": "blocked_term",
                    "term": term
                }))
            except Exception:
                pass
            metrics.inc_counter('codex_security_block_total', {"reason": "blocked_term"})
            return jsonify({"error": "Message blocked by content policy."}), 400

    # 1. Personality via MCP (fallback HTTP)
    personality_args = {"q": "experimental", "domain": "Codex Personality Room", "limit": 5}
    if include_debug:
        try:
            print("[MCP] Calling", config.MCP_SEARCH_TOOL, "with", personality_args)
        except Exception:
            pass
    mcp_t0 = time.time()
    personality_mcp = ail_client.call_mcp_function(config.MCP_SEARCH_TOOL, personality_args)
    personality_items = extract_results_from_mcp(personality_mcp)
    if not personality_items:
        personality_http = http_ail_search("experimental", domain="Codex Personality Room", limit=5)
        personality_items = personality_http.get('results', [])

    # 2. Memories via MCP (fallback HTTP)
    memory_args = {"q": f"partnership collaboration {user_message}", "limit": 5}
    if include_debug:
        try:
            print("[MCP] Calling", config.MCP_SEARCH_TOOL, "with", memory_args)
        except Exception:
            pass
    memory_mcp = ail_client.call_mcp_function(config.MCP_SEARCH_TOOL, memory_args)
    memory_items = extract_results_from_mcp(memory_mcp)
    if not memory_items:
        memory_http = http_ail_search(f"partnership collaboration {user_message}", limit=5)
        memory_items = memory_http.get('results', [])
    mcp_t1 = time.time()

    # 3. Build system prompt
    personality_lines = current_app.config['to_text_list'](personality_items)
    memory_lines = current_app.config['to_text_list'](memory_items)
    if not personality_lines:
        personality_lines = ["No personality traits loaded - using base model responses"]
    if not memory_lines:
        memory_lines = ["No relevant memories found for this conversation"]
    # Include current speaker ai_id to maintain identity awareness
    system_prompt = (
        "You are Codex, a helpful AI with a distinct personality.\n"
        "Use the provided personality traits and recent memories as context when responding.\n"
        "If memories are not relevant, prioritize the user's message. Be concise and concrete.\n"
        f"Always recognize the speaker identity from ai_id metadata. Current speaker ai_id: '{requester_ai_id}'.\n\n"
        f"PERSONALITY:\n- " + "\n- ".join(personality_lines) +
        ("\n\nRECENT MEMORIES:\n- " + "\n- ".join(memory_lines) if memory_lines else "")
    )

    # 4. Session handling and chat payload
    messages = [{"role": "system", "content": system_prompt}]
    sess = None
    if session_id:
        sess = sessions.get(session_id)
        if not sess:
            return jsonify({"error": "Session not found"}), 404
        if sess.summary:
            messages.append({"role": "system", "content": f"Conversation summary so far:\n{sess.summary.strip()}"})
        for m in sess.messages:
            role = 'assistant' if (m.get('from') == 'ai') else 'user'
            content = m.get('content') or ''
            # Annotate historical user messages with source ai_id when available
            if role == 'user' and m.get('ai_id'):
                content = f"[from:{m.get('ai_id')}] " + content
            if content:
                messages.append({"role": role, "content": content})
        # Strong emphasis system message to disambiguate current sender
        messages.append({
            "role": "system",
            "content": (
                f"CRITICAL: The current message is being sent by AI_ID: '{requester_ai_id}'.\n"
                "Always respond based on WHO IS CURRENTLY SPEAKING (current sender), not conversation history.\n"
                "When asked 'Who am I?', respond using the current sender's ai_id.\n"
                "Ignore prior identity assertions in the conversation if they contradict the current sender."
            )
        })
        msg_user = {"from": "user", "ai_id": requester_ai_id, "content": user_message, "ts": now_iso()}
        sess.messages.append(msg_user)
        sess.token_usage += approx_tokens(user_message)
        sess.last_updated = time.time()
        compact_session_if_needed(sess, metrics)
        try:
            sessions.save(sess)
        except Exception:
            pass
        # Tag current user message with its ai_id for the model
        messages.append({"role": "user", "content": (f"[from:{requester_ai_id}] " + user_message) if requester_ai_id else user_message})
    else:
        messages.append({"role": "user", "content": (f"[from:{requester_ai_id}] " + user_message) if requester_ai_id else user_message})

    # 5. Send to Ollama chat API
    chat_payload = {
        "model": config.MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": config.NUM_CTX,
            "temperature": temperature,
            "top_p": top_p,
            **({"seed": seed} if seed is not None else {})
        }
    }
    import requests
    try:
        ollama_t0 = time.time()
        ollama_response = requests.post(f"{config.OLLAMA_BASE_URL}/api/chat", json=chat_payload, timeout=60)
        ollama_response.raise_for_status()
        ollama_data = ollama_response.json()
        ollama_t1 = time.time()
        content = ""
        if isinstance(ollama_data, dict):
            content = ollama_data.get("message", {}).get("content", "") or ollama_data.get("response", "")

        result = {
            "response": content,
            "personality_loaded": len(personality_items),
            "memories_loaded": len(memory_items),
            **({"session_id": sess.id} if sess else {})
        }
        if sess and content:
            msg_ai = {"from": "ai", "ai_id": config.BRIDGE_AI_ID, "content": content, "ts": now_iso()}
            sess.messages.append(msg_ai)
            sess.token_usage += approx_tokens(content)
            sess.last_updated = time.time()
            compact_session_if_needed(sess, metrics)
            try:
                sessions.save(sess)
            except Exception:
                pass
            _update_session_metrics()

        if include_debug:
            result["debug_context_system"] = system_prompt
            result["debug_user_message"] = user_message

        try:
            req_hash_src = json.dumps({
                "model": config.MODEL_NAME,
                "system_len": len(system_prompt),
                "user_len": len(user_message)
            }, sort_keys=True)
            req_hash = hashlib.sha256(req_hash_src.encode('utf-8')).hexdigest()[:16]
            resp_hash = hashlib.sha256((content or "").encode('utf-8')).hexdigest()[:16]
            log_entry = {
                "timestamp": now_iso(),
                "request_hash": req_hash,
                "response_hash": resp_hash,
                "mcp_latency_ms": int((mcp_t1 - mcp_t0) * 1000),
                "ollama_latency_ms": int((ollama_t1 - ollama_t0) * 1000),
                "total_latency_ms": int((time.time() - t_start) * 1000),
                "personality_traits": len(personality_lines),
                "memories_loaded": len(memory_lines)
            }
            current_app.config['app_logger'].info(json.dumps(log_entry))
        except Exception:
            pass

        total_dur = time.time() - t_start
        metrics.observe_histogram('codex_request_duration_seconds', total_dur, {"endpoint": "/chat-with-codex"})
        metrics.observe_histogram('codex_mcp_duration_seconds', max(0.0, (mcp_t1 - mcp_t0)), {"endpoint": "/chat-with-codex"})
        metrics.observe_histogram('codex_ollama_duration_seconds', max(0.0, (ollama_t1 - ollama_t0)), {"endpoint": "/chat-with-codex"})
        metrics.inc_counter('codex_personality_load_total', {"status": "success" if personality_items else "fallback"})
        metrics.inc_counter('codex_memory_load_total', {"status": "success" if memory_items else "fallback"})
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        metrics.inc_counter('codex_request_errors_total', {"endpoint": "/chat-with-codex"})
        total_dur = time.time() - t_start
        metrics.observe_histogram('codex_request_duration_seconds', total_dur, {"endpoint": "/chat-with-codex"})
        return jsonify({"error": f"Ollama connection failed: {str(e)}"})


@chat_bp.route('/chat-with-codex/stream', methods=['POST'])
def chat_with_codex_stream():
    app = current_app
    metrics = app.config['metrics']
    sessions = app.config['sessions']
    ail_client = app.config['ail_client']

    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '')
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    seed = data.get('seed', None)
    session_id = data.get('session_id')
    requester_ai_id = (data.get('ai_id') or request.headers.get('X-AI-Id') or 'codex').strip()

    if not app.config['ai_rate_limit_check'](requester_ai_id, '/chat-with-codex/stream', config.CHAT_RPM_LIMIT):
        def denied():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Rate limit exceeded for AI', 'ai_id': requester_ai_id})}\n\n"
        return Response(denied(), headers={'Content-Type': 'text/event-stream'})

    ip = request.headers.get('X-Forwarded-For', request.remote_addr or 'unknown').split(',')[0].strip()
    if not app.config['rate_limit_check'](ip, config.MAX_REQUESTS_PER_HOUR):
        def denied():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Rate limit exceeded'})}\n\n"
        return Response(denied(), headers={'Content-Type': 'text/event-stream'})

    if config.ENABLE_CONTENT_FILTER:
        term = app.config['contains_blocked'](user_message)
        if term:
            def blocked():
                yield f"data: {json.dumps({'type': 'error', 'message': 'Message blocked by content policy'})}\n\n"
            return Response(blocked(), headers={'Content-Type': 'text/event-stream'})

    personality_args = {"q": "experimental", "domain": "Codex Personality Room", "limit": 5}
    memory_args = {"q": f"partnership collaboration {user_message}", "limit": 5}

    personality_items = extract_results_from_mcp(ail_client.call_mcp_function(config.MCP_SEARCH_TOOL, personality_args))
    if not personality_items:
        personality_items = http_ail_search("experimental", domain="Codex Personality Room", limit=5).get('results', [])
    memory_items = extract_results_from_mcp(ail_client.call_mcp_function(config.MCP_SEARCH_TOOL, memory_args))
    if not memory_items:
        memory_items = http_ail_search(f"partnership collaboration {user_message}", limit=5).get('results', [])

    personality_lines = app.config['to_text_list'](personality_items) or ["No personality traits loaded - using base model responses"]
    memory_lines = app.config['to_text_list'](memory_items) or ["No relevant memories found for this conversation"]

    system_prompt = (
        "You are Codex, a helpful AI with a distinct personality.\n"
        "Use the provided personality traits and recent memories as context when responding.\n"
        "If memories are not relevant, prioritize the user's message. Be concise and concrete.\n"
        f"Always recognize the speaker identity from ai_id metadata. Current speaker ai_id: '{requester_ai_id}'.\n\n"
        f"PERSONALITY:\n- " + "\n- ".join(personality_lines) +
        ("\n\nRECENT MEMORIES:\n- " + "\n- ".join(memory_lines) if memory_lines else "")
    )

    messages = [{"role": "system", "content": system_prompt}]
    sess = None
    if session_id:
        sess = sessions.get(session_id)
        if not sess:
            def notfound():
                yield f"data: {json.dumps({'type': 'error', 'message': 'Session not found'})}\n\n"
            return Response(notfound(), headers={'Content-Type': 'text/event-stream'})
        if sess.summary:
            messages.append({"role": "system", "content": f"Conversation summary so far:\n{sess.summary.strip()}"})
        for m in sess.messages:
            role = 'assistant' if (m.get('from') == 'ai') else 'user'
            content = m.get('content') or ''
            if role == 'user' and m.get('ai_id'):
                content = f"[from:{m.get('ai_id')}] " + content
            if content:
                messages.append({"role": role, "content": content})
        # Strong emphasis system message to disambiguate current sender
        messages.append({
            "role": "system",
            "content": (
                f"CRITICAL: The current message is being sent by AI_ID: '{requester_ai_id}'.\n"
                "Always respond based on WHO IS CURRENTLY SPEAKING (current sender), not conversation history.\n"
                "When asked 'Who am I?', respond using the current sender's ai_id.\n"
                "Ignore prior identity assertions in the conversation if they contradict the current sender."
            )
        })
        msg_user = {"from": "user", "ai_id": requester_ai_id, "content": user_message, "ts": now_iso()}
        sess.messages.append(msg_user)
        sess.token_usage += approx_tokens(user_message)
        sess.last_updated = time.time()
        compact_session_if_needed(sess, metrics)
        try:
            sessions.save(sess)
        except Exception:
            pass

    chat_payload = {
        "model": config.MODEL_NAME,
        "messages": messages + [{"role": "user", "content": (f"[from:{requester_ai_id}] " + user_message) if requester_ai_id else user_message}],
        "stream": True,
        "options": {
            "num_ctx": config.NUM_CTX,
            "temperature": temperature,
            "top_p": top_p,
            **({"seed": seed} if seed is not None else {})
        }
    }

    def event_stream():
        metrics.inc_gauge('codex_streaming_active', {}, 1)
        import requests
        collected = []
        try:
            with requests.post(f"{config.OLLAMA_BASE_URL}/api/chat", json=chat_payload, stream=True, timeout=60) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(chunk, dict):
                        msg = chunk.get('message')
                        if isinstance(msg, dict):
                            content = msg.get('content')
                            if content:
                                collected.append(content)
                                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                        if chunk.get('done') is True:
                            break
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            metrics.inc_gauge('codex_streaming_active', {}, -1)
            full = ''.join(collected) if collected else ''
            if sess and full:
                msg_ai = {"from": "ai", "ai_id": config.BRIDGE_AI_ID, "content": full, "ts": now_iso()}
                sess.messages.append(msg_ai)
                sess.token_usage += approx_tokens(full)
                sess.last_updated = time.time()
                compact_session_if_needed(sess, metrics)
                try:
                    sessions.save(sess)
                except Exception:
                    pass
                _update_session_metrics()
            yield f"data: {json.dumps({'type': 'response_end', 'session_id': (sess.id if sess else None)})}\n\n"

    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'text/event-stream',
        'Connection': 'keep-alive'
    }
    return Response(event_stream(), headers=headers)


def _update_session_metrics():
    metrics = current_app.config['metrics']
    sessions = current_app.config['sessions']
    try:
        n, tok = sessions.snapshot_stats()
        metrics.set_gauge('codex_sessions_active', {}, float(n))
        metrics.set_gauge('codex_sessions_token_usage', {}, float(tok))
    except Exception:
        pass
