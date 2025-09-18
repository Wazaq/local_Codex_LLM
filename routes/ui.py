from flask import Blueprint, render_template

from core.middleware import track_request_metrics

ui_bp = Blueprint('ui', __name__)


@ui_bp.route('/')
@track_request_metrics
def landing_page():
    return render_template('index.html')


@ui_bp.route('/ui/chat')
@track_request_metrics
def serve_chat_ui():
    return render_template('chat.html')
