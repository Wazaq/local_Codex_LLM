import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from flask import Blueprint, current_app, jsonify, render_template, request

from core.middleware import track_request_metrics

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


@admin_bp.route('/dashboard')
@track_request_metrics
def dashboard() -> str:
    return render_template('admin/dashboard.html')


@admin_bp.route('/api/health')
@track_request_metrics
def health_status():
    metrics = current_app.config.get('dashboard_metrics')
    if metrics is None:
        return jsonify({'status': 'error', 'error': 'Metrics collector unavailable'}), 500

    summary = metrics.get_summary()
    sessions_info = _gather_session_summary()
    health = _calculate_health(summary['error_rate_percent'])

    avg_times = summary.get('average_response_times') or {}
    avg_ms = 0.0
    if avg_times:
        avg_ms = round(sum(avg_times.values()) / len(avg_times) * 1000, 2)

    return jsonify({
        'status': health,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'uptime': summary['uptime_formatted'],
        'metrics': {
            'total_requests': summary['total_requests'],
            'total_errors': summary['total_errors'],
            'error_rate_percent': summary['error_rate_percent'],
            'average_response_time_ms': avg_ms,
            'average_response_times': {
                endpoint: round(value * 1000, 2)
                for endpoint, value in (avg_times or {}).items()
            },
            'counter_count': len(summary.get('active_counters', {})),
        },
        'sessions': sessions_info,
        'system': {
            'start_time': metrics.start_time.isoformat(),
            'storage': sessions_info.get('storage'),
            'storage_accessible': sessions_info.get('storage_accessible'),
        },
    })


@admin_bp.route('/api/metrics')
@track_request_metrics
def metrics_api():
    hours_back = request.args.get('hours', default=24, type=int)
    metrics = current_app.config.get('dashboard_metrics')
    if metrics is None:
        return jsonify({'error': 'Metrics collector unavailable'}), 500

    data = metrics.get_time_series_data(hours_back)
    summary = metrics.get_summary()
    return jsonify({
        'summary': summary,
        'time_series': data,
        'generated_at': datetime.now(timezone.utc).isoformat(),
    })


@admin_bp.route('/api/sessions')
@track_request_metrics
def sessions_api():
    sessions = _collect_sessions()
    sort_key = lambda item: item['last_activity'] or ''
    sessions.sort(key=sort_key, reverse=True)
    return jsonify({
        'sessions': sessions,
        'total_count': len(sessions),
    })


@admin_bp.route('/api/errors')
@track_request_metrics
def errors_api():
    hours_back = request.args.get('hours', default=24, type=int)
    metrics = current_app.config.get('dashboard_metrics')
    if metrics is None:
        return jsonify({'error': 'Metrics collector unavailable'}), 500

    series = metrics.get_time_series_data(hours_back)
    recent_errors = series.get('errors', [])

    grouped: Dict[str, Dict[str, Any]] = {}
    for error in recent_errors:
        code = error.get('error_code', 'UNKNOWN')
        group = grouped.setdefault(code, {
            'count': 0,
            'latest_message': '',
            'latest_timestamp': '',
            'endpoints': set(),
        })
        group['count'] += 1
        timestamp = error.get('timestamp', '')
        if timestamp >= group['latest_timestamp']:
            group['latest_timestamp'] = timestamp
            group['latest_message'] = error.get('message', '')
        endpoint = error.get('endpoint')
        if endpoint:
            group['endpoints'].add(endpoint)

    serialized = {
        code: {
            'count': data['count'],
            'latest_message': data['latest_message'],
            'latest_timestamp': data['latest_timestamp'],
            'endpoints': sorted(data['endpoints']),
        }
        for code, data in grouped.items()
    }

    return jsonify({
        'total_errors': len(recent_errors),
        'error_groups': serialized,
        'generated_at': datetime.now(timezone.utc).isoformat(),
    })


def _collect_sessions() -> List[Dict[str, Any]]:
    store = current_app.config.get('sessions')
    if store is None:
        return []

    sessions: List[Dict[str, Any]] = []
    try:
        ids = store.list_ids()
    except Exception:
        return []

    for sid in ids:
        session = store.get(sid)
        if session is None:
            continue

        last_updated = getattr(session, 'last_updated', None)
        last_iso = _timestamp_to_iso(last_updated)
        sessions.append({
            'session_id': getattr(session, 'id', sid),
            'ai_ids': list(getattr(session, 'ai_ids', []) or []),
            'created_at': getattr(session, 'created_at', None),
            'last_activity': last_iso,
            'message_count': len(getattr(session, 'messages', []) or []),
            'token_usage': getattr(session, 'token_usage', 0),
        })

    return sessions


def _gather_session_summary() -> Dict[str, Any]:
    store = current_app.config.get('sessions')
    sessions = _collect_sessions()
    active_cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    active = 0
    total_messages = 0
    for session in sessions:
        total_messages += session['message_count']
        last = session.get('last_activity')
        if last and _iso_to_datetime(last) >= active_cutoff:
            active += 1

    storage_meta = _session_store_metadata(store)

    return {
        'total': len(sessions),
        'active': active,
        'total_messages': total_messages,
        **storage_meta,
    }


def _session_store_metadata(store: Any) -> Dict[str, Any]:
    if store is None:
        return {'storage': 'unknown', 'storage_accessible': False}

    cls = type(store).__name__.lower()
    metadata: Dict[str, Any] = {
        'storage': 'memory',
        'storage_accessible': True,
    }
    if 'file' in cls:
        metadata['storage'] = 'file'
        path = getattr(store, '_path', None)
        metadata['path'] = path
        if path:
            directory = os.path.dirname(path) or '.'
            metadata['storage_accessible'] = os.access(directory, os.R_OK | os.W_OK)
    elif 'redis' in cls:
        metadata['storage'] = 'redis'
    else:
        metadata['storage'] = cls or 'memory'
    return metadata


def _timestamp_to_iso(timestamp: Any) -> str:
    if not timestamp:
        return ''
    try:
        return datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat()
    except Exception:
        return ''


def _iso_to_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except Exception:
        return datetime.fromtimestamp(0, timezone.utc)


def _calculate_health(error_rate_percent: float) -> str:
    if error_rate_percent > 25:
        return 'unhealthy'
    if error_rate_percent > 10:
        return 'degraded'
    return 'healthy'
