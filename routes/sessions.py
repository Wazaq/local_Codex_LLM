from datetime import datetime, timezone

from flask import Blueprint, request, jsonify, current_app

from core.session_store import approx_tokens, compact_session_if_needed
from core.errors import (
    CodexError,
    ErrorCode,
    create_error_response,
    invalid_request_error,
    session_not_found_error,
)
from core.logging_config import log_error, log_request


sessions_bp = Blueprint('sessions', __name__)


@sessions_bp.route('/sessions', methods=['POST'])
def create_session():
    logger = current_app.logger
    log_request(logger, '/sessions')

    metrics = current_app.config['metrics']
    sessions = current_app.config['sessions']
    data = request.get_json(silent=True)
    if data is None:
        return invalid_request_error(
            "Request body must be valid JSON",
            {"endpoint": "/sessions"}
        )

    ai_ids = data.get('ai_ids') or []
    if not isinstance(ai_ids, list):
        return invalid_request_error(
            "Invalid 'ai_ids' parameter",
            {"expected": "array", "received": type(ai_ids).__name__}
        )

    import config
    try:
        max_tokens_value = data.get('max_tokens') or config.SESSION_TOKEN_LIMIT
        max_tokens = int(max_tokens_value)
    except (TypeError, ValueError):
        error_payload, status = create_error_response(
            ErrorCode.INVALID_PARAMETERS,
            "Invalid 'max_tokens' value",
            {"received": data.get('max_tokens')},
            "Provide an integer value for 'max_tokens'.",
            status_code=400,
        )
        return error_payload, status

    ttl_seconds = data.get('ttl_seconds')
    display_name = data.get('display_name')
    try:
        s = sessions.create(ai_ids=ai_ids, max_tokens=max_tokens, ttl_seconds=ttl_seconds, display_name=display_name)
    except CodexError as exc:
        log_error(logger, exc, {"endpoint": "/sessions"})
        return exc.to_dict(), exc.status_code
    except Exception as exc:
        log_error(logger, exc, {"endpoint": "/sessions"})
        error_payload, status = create_error_response(
            ErrorCode.SYSTEM_ERROR,
            "Failed to create session",
            {"error_type": type(exc).__name__},
            "Try again later.",
            status_code=500,
        )
        return error_payload, status

    metrics.inc_counter('codex_sessions_created_total', {})
    _update_session_metrics()
    return jsonify({"ok": True, "session": s.to_dict(False)})


@sessions_bp.route('/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    logger = current_app.logger
    log_request(logger, f'/sessions/{session_id}', session_id=session_id)

    sessions = current_app.config['sessions']
    try:
        s = sessions.get(session_id)
        if not s:
            return session_not_found_error(session_id)
        return jsonify({"ok": True, "session": s.to_dict(False)})
    except Exception as exc:
        log_error(logger, exc, {"endpoint": "/sessions/<session_id>", "session_id": session_id})
        error_payload, status = create_error_response(
            ErrorCode.SYSTEM_ERROR,
            "Failed to retrieve session",
            {"session_id": session_id},
            "Check the session ID and retry.",
            status_code=500,
        )
        return error_payload, status


@sessions_bp.route('/sessions/<session_id>/messages', methods=['GET'])
def get_session_messages(session_id):
    logger = current_app.logger
    log_request(logger, f'/sessions/{session_id}/messages', session_id=session_id)

    sessions = current_app.config['sessions']
    try:
        s = sessions.get(session_id)
        if not s:
            return session_not_found_error(session_id)

        limit = request.args.get('limit')
        msgs = list(s.messages)
        if limit:
            try:
                n = int(limit)
                if n >= 0:
                    msgs = msgs[-n:]
            except (TypeError, ValueError):
                return invalid_request_error(
                    "Invalid 'limit' parameter",
                    {"received": limit, "session_id": session_id}
                )

        return jsonify({
            "ok": True,
            "session_id": s.id,
            "display_name": s.display_name,
            "summary": s.summary,
            "messages": msgs,
            "token_usage": s.token_usage
        })
    except CodexError as exc:
        log_error(logger, exc, {"endpoint": "/sessions/<session_id>/messages", "session_id": session_id})
        return exc.to_dict(), exc.status_code
    except Exception as exc:
        log_error(logger, exc, {"endpoint": "/sessions/<session_id>/messages", "session_id": session_id})
        error_payload, status = create_error_response(
            ErrorCode.SYSTEM_ERROR,
            "Failed to retrieve session messages",
            {"session_id": session_id, "error_type": type(exc).__name__},
            "Please try again later.",
            status_code=500,
        )
        return error_payload, status


@sessions_bp.route('/sessions', methods=['GET'])
def list_sessions():
    logger = current_app.logger
    log_request(logger, '/sessions')

    sessions = current_app.config['sessions']
    try:
        ids = sessions.list_ids()
        out = []
        for sid in ids:
            s = sessions.get(sid)
            if not s:
                continue
            out.append({
                "session_id": s.id,
                "display_name": s.display_name,
                "created_at": s.created_at,
                "messages": len(s.messages or []),
                "message_count": len(s.messages or []),
                "token_usage": s.token_usage,
                "ai_ids": s.ai_ids,
                "summary_len": len(s.summary or ''),
                "last_activity": _ts_to_iso(s.last_updated),
            })
        return jsonify({
            "ok": True,
            "success": True,
            "count": len(out),
            "sessions": out
        })
    except Exception as exc:
        log_error(logger, exc, {"endpoint": "/sessions"})
        error_payload, status = create_error_response(
            ErrorCode.SYSTEM_ERROR,
            "Failed to list sessions",
            {"error_type": type(exc).__name__},
            "Please try again later.",
            status_code=500,
        )
        return error_payload, status


@sessions_bp.route('/sessions/<session_id>/messages', methods=['POST'])
def add_session_message(session_id):
    logger = current_app.logger
    log_request(logger, f'/sessions/{session_id}/messages', session_id=session_id)

    sessions = current_app.config['sessions']
    metrics = current_app.config['metrics']

    try:
        s = sessions.get(session_id)
        if not s:
            return session_not_found_error(session_id)

        data = request.get_json(silent=True)
        if data is None:
            return invalid_request_error(
                "Request body must be valid JSON",
                {"endpoint": "/sessions/<session_id>/messages", "session_id": session_id}
            )

        frm = (data.get('from') or '').strip() or 'user'
        content = data.get('content') or ''
        ai_id = (data.get('ai_id') or '').strip()

        if ai_id and s.ai_ids and ai_id not in s.ai_ids:
            error_payload, status = create_error_response(
                ErrorCode.INVALID_PARAMETERS,
                "AI not permitted in this session",
                {"ai_id": ai_id, "session_id": session_id},
                "Use an allowed AI ID for this session.",
                status_code=403,
            )
            return error_payload, status

        msg = {"from": frm, "content": content, **({"ai_id": ai_id} if ai_id else {}), "ts": current_app.config['now_iso']()}
        s.messages.append(msg)
        s.token_usage += approx_tokens(content)
        import time as _t
        s.last_updated = _t.time()
        compact_session_if_needed(s, metrics)
        try:
            sessions.save(s)
        except Exception as exc:
            log_error(logger, exc, {"endpoint": "/sessions/<session_id>/messages", "session_id": session_id})
        _update_session_metrics()
        return jsonify({"ok": True, "session": s.to_dict(False)})

    except CodexError as exc:
        log_error(logger, exc, {"endpoint": "/sessions/<session_id>/messages", "session_id": session_id})
        return exc.to_dict(), exc.status_code
    except Exception as exc:
        log_error(logger, exc, {"endpoint": "/sessions/<session_id>/messages", "session_id": session_id})
        error_payload, status = create_error_response(
            ErrorCode.SYSTEM_ERROR,
            "Failed to add session message",
            {"session_id": session_id, "error_type": type(exc).__name__},
            "Please try again later.",
            status_code=500,
        )
        return error_payload, status


@sessions_bp.route('/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    logger = current_app.logger
    log_request(logger, f'/sessions/{session_id}', session_id=session_id)

    sessions = current_app.config['sessions']
    metrics = current_app.config['metrics']
    try:
        ok = sessions.delete(session_id)
        if not ok:
            return session_not_found_error(session_id)
        metrics.inc_counter('codex_sessions_deleted_total', {})
        _update_session_metrics()
        return jsonify({"ok": True})
    except CodexError as exc:
        log_error(logger, exc, {"endpoint": "/sessions/<session_id>", "session_id": session_id})
        return exc.to_dict(), exc.status_code
    except Exception as exc:
        log_error(logger, exc, {"endpoint": "/sessions/<session_id>", "session_id": session_id})
        error_payload, status = create_error_response(
            ErrorCode.SYSTEM_ERROR,
            "Failed to delete session",
            {"session_id": session_id, "error_type": type(exc).__name__},
            "Please try again later.",
            status_code=500,
        )
        return error_payload, status


@sessions_bp.route('/sessions/<session_id>', methods=['PATCH'])
def update_session(session_id):
    logger = current_app.logger
    log_request(logger, f'/sessions/{session_id}', session_id=session_id)

    sessions = current_app.config['sessions']

    try:
        s = sessions.get(session_id)
        if not s:
            return session_not_found_error(session_id)

        data = request.get_json(silent=True)
        if not data:
            return invalid_request_error(
                "Request body must be valid JSON",
                {"endpoint": "/sessions/<session_id>", "session_id": session_id}
            )

        if 'display_name' not in data:
            return invalid_request_error(
                "Missing 'display_name' field",
                {"endpoint": "/sessions/<session_id>", "session_id": session_id}
            )

        display_name = data.get('display_name')
        if display_name is not None and not isinstance(display_name, str):
            return invalid_request_error(
                "Invalid 'display_name' type",
                {"expected": "string", "received": type(display_name).__name__}
            )

        s.display_name = (display_name or '').strip() or None
        try:
            sessions.save(s)
        except Exception:
            pass
        return jsonify({"ok": True, "session": s.to_dict(False)})

    except CodexError as exc:
        log_error(logger, exc, {"endpoint": "/sessions/<session_id>", "session_id": session_id})
        return exc.to_dict(), exc.status_code
    except Exception as exc:
        log_error(logger, exc, {"endpoint": "/sessions/<session_id>", "session_id": session_id})
        error_payload, status = create_error_response(
            ErrorCode.SYSTEM_ERROR,
            "Failed to update session",
            {"session_id": session_id, "error_type": type(exc).__name__},
            "Please try again later.",
            status_code=500,
        )
        return error_payload, status


def _update_session_metrics():
    metrics = current_app.config['metrics']
    sessions = current_app.config['sessions']
    try:
        n, tok = sessions.snapshot_stats()
        metrics.set_gauge('codex_sessions_active', {}, float(n))
        metrics.set_gauge('codex_sessions_token_usage', {}, float(tok))
    except Exception:
        pass


def _ts_to_iso(ts):
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None
