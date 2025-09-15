from flask import Blueprint, request, jsonify, current_app

from core.session_store import approx_tokens, compact_session_if_needed


sessions_bp = Blueprint('sessions', __name__)


@sessions_bp.route('/sessions', methods=['POST'])
def create_session():
    metrics = current_app.config['metrics']
    sessions = current_app.config['sessions']
    data = request.get_json(silent=True) or {}
    ai_ids = data.get('ai_ids') or []
    import config
    max_tokens = int(data.get('max_tokens') or config.SESSION_TOKEN_LIMIT)
    ttl_seconds = data.get('ttl_seconds')
    s = sessions.create(ai_ids=ai_ids, max_tokens=max_tokens, ttl_seconds=ttl_seconds)
    metrics.inc_counter('codex_sessions_created_total', {})
    _update_session_metrics()
    return jsonify({"ok": True, "session": s.to_dict(False)})


@sessions_bp.route('/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    sessions = current_app.config['sessions']
    s = sessions.get(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"ok": True, "session": s.to_dict(False)})


@sessions_bp.route('/sessions/<session_id>/messages', methods=['GET'])
def get_session_messages(session_id):
    sessions = current_app.config['sessions']
    s = sessions.get(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    limit = request.args.get('limit')
    msgs = list(s.messages)
    if limit:
        try:
            n = int(limit)
            if n >= 0:
                msgs = msgs[-n:]
        except Exception:
            pass
    return jsonify({"ok": True, "session_id": s.id, "summary": s.summary, "messages": msgs, "token_usage": s.token_usage})


@sessions_bp.route('/sessions', methods=['GET'])
def list_sessions():
    sessions = current_app.config['sessions']
    ids = sessions.list_ids()
    out = []
    for sid in ids:
        s = sessions.get(sid)
        if not s:
            continue
        out.append({
            "session_id": s.id,
            "created_at": s.created_at,
            "messages": len(s.messages or []),
            "token_usage": s.token_usage,
            "ai_ids": s.ai_ids,
            "summary_len": len(s.summary or '')
        })
    return jsonify({"ok": True, "sessions": out})


@sessions_bp.route('/sessions/<session_id>/messages', methods=['POST'])
def add_session_message(session_id):
    sessions = current_app.config['sessions']
    metrics = current_app.config['metrics']
    s = sessions.get(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    data = request.get_json(silent=True) or {}
    frm = (data.get('from') or '').strip() or 'user'
    content = data.get('content') or ''
    ai_id = (data.get('ai_id') or '').strip()
    if ai_id and s.ai_ids and ai_id not in s.ai_ids:
        return jsonify({"error": "AI not permitted in this session", "ai_id": ai_id}), 403
    msg = {"from": frm, "content": content, **({"ai_id": ai_id} if ai_id else {}), "ts": current_app.config['now_iso']()}
    s.messages.append(msg)
    s.token_usage += approx_tokens(content)
    import time as _t
    s.last_updated = _t.time()
    compact_session_if_needed(s, metrics)
    try:
        sessions.save(s)
    except Exception:
        pass
    _update_session_metrics()
    return jsonify({"ok": True, "session": s.to_dict(False)})


@sessions_bp.route('/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    sessions = current_app.config['sessions']
    metrics = current_app.config['metrics']
    ok = sessions.delete(session_id)
    if not ok:
        return jsonify({"error": "Session not found"}), 404
    metrics.inc_counter('codex_sessions_deleted_total', {})
    _update_session_metrics()
    return jsonify({"ok": True})


def _update_session_metrics():
    metrics = current_app.config['metrics']
    sessions = current_app.config['sessions']
    try:
        n, tok = sessions.snapshot_stats()
        metrics.set_gauge('codex_sessions_active', {}, float(n))
        metrics.set_gauge('codex_sessions_token_usage', {}, float(tok))
    except Exception:
        pass
