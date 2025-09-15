from flask import Blueprint, render_template

ui_bp = Blueprint('ui', __name__)


@ui_bp.route('/ui/chat')
def serve_chat_ui():
    return render_template('chat.html')

