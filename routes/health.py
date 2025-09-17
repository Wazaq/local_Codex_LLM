from flask import Blueprint, jsonify, Response, current_app
import time
import os

import config
from core.errors import ErrorCode, create_error_response
from core.logging_config import log_error, log_request

health_bp = Blueprint('health', __name__)


def _check_dependencies(app):
    metrics = app.config['metrics']
    ail_client = app.config['ail_client']
    # MCP
    mcp_ok = False
    mcp_tools = 0
    mcp_rt = None
    try:
        t0 = time.time()
        resp = ail_client.list_tools()
        mcp_rt = int((time.time() - t0) * 1000)
        if isinstance(resp, dict) and 'error' not in resp:
            result = resp.get('result', resp)
            tools = result.get('tools') if isinstance(result, dict) else None
            if isinstance(tools, list):
                mcp_tools = len(tools)
                mcp_ok = True
    except Exception:
        pass
    # Ollama
    ollama_ok = False
    ollama_rt = None
    import requests
    try:
        t0 = time.time()
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        ollama_rt = int((time.time() - t0) * 1000)
        if r.ok:
            ollama_ok = True
    except Exception:
        pass
    try:
        metrics.set_gauge('codex_health_status', {"component": "mcp"}, 1 if mcp_ok else 0)
        metrics.set_gauge('codex_health_status', {"component": "ollama"}, 1 if ollama_ok else 0)
    except Exception:
        pass
    return mcp_ok, mcp_tools, mcp_rt, ollama_ok, ollama_rt


@health_bp.route('/health', methods=['GET'])
def health_check():
    logger = current_app.logger
    log_request(logger, '/health')

    try:
        ail_client = current_app.config['ail_client']
        sessions = current_app.config.get('sessions')
        count, _ = (sessions.snapshot_stats() if sessions else (0, 0))
        storage_info = {
            "storage": "memory",
            "backend": "memory",
            "storage_accessible": True,
        }
        try:
            if sessions is not None:
                cls = type(sessions).__name__.lower()
                if 'file' in cls:
                    storage_info.update({
                        "storage": "file-based",
                        "backend": "file",
                    })
                    path = getattr(sessions, '_path', getattr(config, 'SESSION_STORAGE_PATH', 'data/sessions.json'))
                    d = os.path.dirname(path) or '.'
                    storage_info['path'] = path
                    storage_info['storage_accessible'] = bool(os.access(d, os.R_OK | os.W_OK))
                elif 'redis' in cls:
                    storage_info.update({"storage": "redis", "backend": "redis"})
                    storage_info['storage_accessible'] = True
                else:
                    _ = sessions.list_ids()
                    storage_info['storage_accessible'] = True
        except Exception:
            storage_info['storage_accessible'] = False
        mcp_server = current_app.config.get('mcp_server')
        tools_count = 0
        if mcp_server is not None:
            try:
                listed = mcp_server.server.list_tools()
                tools = listed.get('tools') if isinstance(listed, dict) else []
                tools_count = len(tools) if isinstance(tools, list) else 0
            except Exception:
                pass
        storage_info['accessible'] = storage_info.get('storage_accessible')
        return jsonify({
            "status": "MCP Bridge running",
            "ail_url": ail_client.mcp_url,
            "sessions": {"count": count, **storage_info},
            "mcp": {"enabled": mcp_server is not None, "tools_count": tools_count, "endpoint": "/mcp"},
        })
    except Exception as exc:
        log_error(logger, exc, {"endpoint": "/health"})
        error_payload, status = create_error_response(
            ErrorCode.SYSTEM_ERROR,
            "Failed to gather health information",
            {"error_type": type(exc).__name__},
            "Try again later.",
            status_code=500,
        )
        return error_payload, status


@health_bp.route('/health/mcp', methods=['GET'])
def health_mcp():
    logger = current_app.logger
    log_request(logger, '/health/mcp')

    try:
        mcp_ok, mcp_tools, mcp_rt, ollama_ok, ollama_rt = _check_dependencies(current_app)
        model_name = config.MODEL_NAME
        status = 'healthy' if (mcp_ok and ollama_ok) else ('degraded' if (mcp_ok or ollama_ok) else 'unhealthy')
        return jsonify({
            "status": status,
            "mcp": {"connected": mcp_ok, "tools_available": mcp_tools, "response_time_ms": mcp_rt},
            "ollama": {"connected": ollama_ok, "model": model_name, "response_time_ms": ollama_rt},
        })
    except Exception as exc:
        log_error(logger, exc, {"endpoint": "/health/mcp"})
        error_payload, status_code = create_error_response(
            ErrorCode.SYSTEM_ERROR,
            "Failed to gather MCP health information",
            {"error_type": type(exc).__name__},
            "Try again later.",
            status_code=500,
        )
        return error_payload, status_code


@health_bp.route('/healthz', methods=['GET'])
def healthz():
    logger = current_app.logger
    log_request(logger, '/healthz')

    try:
        from core.utils import now_iso
        return jsonify({"status": "alive", "timestamp": now_iso()})
    except Exception as exc:
        log_error(logger, exc, {"endpoint": "/healthz"})
        error_payload, status = create_error_response(
            ErrorCode.SYSTEM_ERROR,
            "Failed to report healthz status",
            {"error_type": type(exc).__name__},
            "Try again later.",
            status_code=500,
        )
        return error_payload, status


@health_bp.route('/readyz', methods=['GET'])
def readyz():
    logger = current_app.logger
    log_request(logger, '/readyz')

    try:
        mcp_ok, _, _, ollama_ok, _ = _check_dependencies(current_app)
        ready = bool(mcp_ok and ollama_ok)
        status_code = 200 if ready else 503
        from core.utils import now_iso
        return jsonify({"ready": ready, "timestamp": now_iso()}), status_code
    except Exception as exc:
        log_error(logger, exc, {"endpoint": "/readyz"})
        error_payload, status_code = create_error_response(
            ErrorCode.SYSTEM_ERROR,
            "Failed to report readiness",
            {"error_type": type(exc).__name__},
            "Try again later.",
            status_code=500,
        )
        return error_payload, status_code


@health_bp.route('/metrics', methods=['GET'])
def metrics_endpoint():
    logger = current_app.logger
    log_request(logger, '/metrics')

    try:
        text = current_app.config['metrics'].render_prometheus()
        return Response(text, mimetype='text/plain; version=0.0.4; charset=utf-8')
    except Exception as exc:
        log_error(logger, exc, {"endpoint": "/metrics"})
        error_payload, status = create_error_response(
            ErrorCode.SYSTEM_ERROR,
            "Failed to render metrics",
            {"error_type": type(exc).__name__},
            "Try again later.",
            status_code=500,
        )
        return error_payload, status


@health_bp.route('/config', methods=['GET'])
def get_config():
    logger = current_app.logger
    log_request(logger, '/config')

    try:
        cfg = {
            "model": config.MODEL_NAME,
            "num_ctx": config.NUM_CTX,
            "mcp_tool": config.MCP_SEARCH_TOOL,
            "log_level": config.LOG_LEVEL,
            "mcp_retry_attempts": config.MCP_RETRY_ATTEMPTS,
            "mcp_timeout_seconds": config.MCP_TIMEOUT_SECONDS,
            "mcp_backoff_base": config.MCP_BACKOFF_BASE,
            "http_fallback_timeout": config.HTTP_FALLBACK_TIMEOUT,
            "circuit_breaker_threshold": config.CIRCUIT_BREAKER_THRESHOLD,
            "circuit_breaker_cooldown": config.CIRCUIT_BREAKER_COOLDOWN,
            "content_filter_enabled": config.ENABLE_CONTENT_FILTER,
            "max_requests_per_hour": config.MAX_REQUESTS_PER_HOUR,
            "allow_list_domains": config.ALLOW_LIST_DOMAINS,
            "security_log_file": config.SECURITY_LOG_FILE,
            "shutdown_timeout": config.SHUTDOWN_TIMEOUT,
            "shutdown_endpoint_enabled": config.ENABLE_SHUTDOWN_ENDPOINT,
        }
        issues = []
        if config.NUM_CTX <= 0:
            issues.append('NUM_CTX must be positive')
        if config.MCP_RETRY_ATTEMPTS < 0:
            issues.append('MCP_RETRY_ATTEMPTS must be >= 0')
        if config.MCP_TIMEOUT_SECONDS <= 0:
            issues.append('MCP_TIMEOUT_SECONDS must be > 0')
        if config.MCP_BACKOFF_BASE <= 0:
            issues.append('MCP_BACKOFF_BASE must be > 0')
        if config.SHUTDOWN_TIMEOUT < 0:
            issues.append('SHUTDOWN_TIMEOUT must be >= 0')
        return jsonify({"config": cfg, "valid": len(issues) == 0, "issues": issues})
    except Exception as exc:
        log_error(logger, exc, {"endpoint": "/config"})
        error_payload, status = create_error_response(
            ErrorCode.SYSTEM_ERROR,
            "Failed to load configuration",
            {"error_type": type(exc).__name__},
            "Please try again later.",
            status_code=500,
        )
        return error_payload, status
