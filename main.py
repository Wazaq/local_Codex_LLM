import json
import logging
import os
import signal
import threading
import time
from logging.handlers import TimedRotatingFileHandler

from flask import Flask, jsonify, request

import config
from core.metrics import PrometheusMetrics
from core.mcp_client import HTTPMCPClient
from core.session_store import make_session_store
from core import security as sec
from core import utils
from core.logging_config import setup_logging
from routes.chat import chat_bp
from routes.sessions import sessions_bp
from routes.ail import ail_bp
from routes.ui import ui_bp
from routes.health import health_bp
from mcp.server import CodexMCPServer


def create_app():
    app = Flask(__name__)

    app_logger = setup_logging(app)
    app_logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    # Initialize core components
    metrics = PrometheusMetrics()
    metrics.set_gauge('codex_streaming_active', {}, 0)
    ail_client = HTTPMCPClient(config.MCP_URL)
    sessions = make_session_store()

    # Add rotating log handler alongside structured logging (rotate daily, keep 7 backups)
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        logs_dir = os.path.dirname(__file__)
    log_path = os.path.normpath(os.path.join(logs_dir, 'codex_bridge.log'))
    if not any(isinstance(handler, TimedRotatingFileHandler) and getattr(handler, 'baseFilename', None) == log_path for handler in app_logger.handlers):
        rotating_handler = TimedRotatingFileHandler(log_path, when='midnight', backupCount=7, encoding='utf-8')
        rotating_handler.setFormatter(logging.Formatter('%(message)s'))
        app_logger.addHandler(rotating_handler)

    security_logger = logging.getLogger('codex_security')
    if not security_logger.handlers:
        try:
            os.makedirs(os.path.dirname(config.SECURITY_LOG_FILE), exist_ok=True)
        except Exception:
            pass
        sh = TimedRotatingFileHandler(config.SECURITY_LOG_FILE, when='midnight', backupCount=14, encoding='utf-8')
        sh.setFormatter(logging.Formatter('%(message)s'))
        security_logger.addHandler(sh)
        security_logger.setLevel(logging.INFO)

    # Dependency injection into app config for route access
    app.config['metrics'] = metrics
    app.config['ail_client'] = ail_client
    app.config['sessions'] = sessions
    app.config['app_logger'] = app_logger
    app.config['security_logger'] = security_logger

    app.config['rate_limit_check'] = sec.rate_limit_check
    app.config['ai_rate_limit_check'] = sec.ai_rate_limit_check
    app.config['contains_blocked'] = sec.contains_blocked
    app.config['now_iso'] = utils.now_iso
    # to_text_list uses content filtering optionally
    app.config['to_text_list'] = lambda items: utils.to_text_list(
        items, metrics=metrics, security_logger=security_logger, content_filter=config.ENABLE_CONTENT_FILTER
    )

    # Register blueprints
    app.register_blueprint(chat_bp)
    app.register_blueprint(sessions_bp)
    app.register_blueprint(ail_bp)
    app.register_blueprint(ui_bp)
    app.register_blueprint(health_bp)

    # MCP server integration (used by legacy /mcp route in routes.ail)
    app.config['mcp_server'] = CodexMCPServer(app)

    # Shutdown gate
    app.config['SHUTTING_DOWN'] = False

    @app.before_request
    def _gate_shutting_down():
        if app.config.get('SHUTTING_DOWN'):
            path = request.path or ''
            if path.startswith('/health') or path.startswith('/metrics'):
                return None
            return jsonify({"error": "Server shutting down"}), 503

    if config.ENABLE_SHUTDOWN_ENDPOINT:
        @app.route('/shutdown', methods=['POST'])
        def shutdown_now():
            _handle_sigterm(None, None, app)
            return jsonify({"status": "shutting_down", "timeout": config.SHUTDOWN_TIMEOUT})

    return app


def _handle_sigterm(signum, frame, app):
    app.config['SHUTTING_DOWN'] = True
    def _exit_later():
        time.sleep(config.SHUTDOWN_TIMEOUT)
        os._exit(0)
    threading.Thread(target=_exit_later, daemon=True).start()


if __name__ == '__main__':
    print("🚀 Starting Codex MCP Bridge (modular)...")
    print(f"🔗 AIL Connection: {config.MCP_URL}")
    print(f"🤖 Ollama Connection: {config.OLLAMA_BASE_URL}")

    app = create_app()

    # Validate config (mirrors monolith logic)
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
    if issues:
        print("⚠️ Config issues:", "; ".join(issues))

    # Optionally start resource monitor if psutil is available
    try:
        import psutil  # type: ignore
        proc = psutil.Process()
        def _resource_monitor():
            while True:
                try:
                    rss = proc.memory_info().rss
                    cpu = proc.cpu_percent(interval=1.0)
                    app.config['metrics'].set_gauge('process_resident_memory_bytes', {}, rss)
                    app.config['metrics'].set_gauge('process_cpu_percent', {}, float(cpu))
                except Exception:
                    time.sleep(5)
                time.sleep(5)
        threading.Thread(target=_resource_monitor, daemon=True).start()
        print("📊 Resource monitor enabled (psutil)")
    except Exception:
        print("ℹ️ psutil not available; resource monitor disabled")

    try:
        signal.signal(signal.SIGTERM, lambda s, f: _handle_sigterm(s, f, app))
        signal.signal(signal.SIGINT, lambda s, f: _handle_sigterm(s, f, app))
    except Exception:
        pass

    print("🌐 Bridge running on: http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
