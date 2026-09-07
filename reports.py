from datetime import datetime, timedelta


AUDIT_EVENT_TYPES = {'login', 'logout', 'create', 'delete', 'settings'}

REPORT_TITLES = {
    'village': 'Village-Wide Security Summary Report',
    'incident': 'Security Incident Report',
    'frequency': 'Alert Frequency Analytics Report',
    'surveillance': 'Surveillance System Health Report',
}


def _severity(event_type):
    normalized = (event_type or '').lower()
    if any(keyword in normalized for keyword in ('critical', 'unknown', 'alert', 'error')):
        return 'High'
    if any(keyword in normalized for keyword in ('motion', 'object', 'detection')):
        return 'Warning'
    return 'Normal'


def _matches_zone(source_name, zone):
    if not zone or zone == 'all':
        return True

    source = (source_name or '').lower()
    zone_aliases = {
        'gate1': ('gate 1', 'gate1', 'cam 1'),
        'north': ('north', 'cam 2'),
        'clubhouse': ('clubhouse', 'amenities', 'cam 3'),
    }
    return any(alias in source for alias in zone_aliases.get(zone, (zone.lower(),)))


def _matches_severity(event_severity, severity):
    if not severity or severity == 'all':
        return True
    if severity == 'critical':
        return event_severity == 'High'
    if severity == 'normal':
        return event_severity == 'Normal'
    return True


def build_report(event_log_model, report_type='village', timeframe='7d', zone='all', severity='all', now=None):
    """Build the admin report context from security events in EventLog."""
    now = now or datetime.utcnow()
    report_type = report_type if report_type in REPORT_TITLES else 'village'
    timeframe = timeframe if timeframe in {'24h', '7d', '30d'} else '7d'
    zone = zone if zone in {'all', 'gate1', 'north', 'clubhouse'} else 'all'
    severity = severity if severity in {'all', 'critical', 'normal'} else 'all'
    timeframe_days = {'24h': 1, '7d': 7, '30d': 30}[timeframe]
    since = now - timedelta(days=timeframe_days)

    events = event_log_model.query.filter(
        event_log_model.timestamp >= since,
        ~event_log_model.event_type.in_(AUDIT_EVENT_TYPES),
    ).order_by(event_log_model.timestamp.desc()).all()

    report_events = []
    for event in events:
        event_severity = _severity(event.event_type)
        if not _matches_zone(event.source_name, zone) or not _matches_severity(event_severity, severity):
            continue
        event_label = event.event_type or 'Unspecified event'
        if event_label.upper() == 'CRITICAL ALERT' and event.description:
            event_label = f'{event_label}: {event.description.removeprefix("Detected: ").strip()}'
        report_events.append({
            'timestamp': event.timestamp.strftime('%Y-%m-%d %I:%M:%S %p') if event.timestamp else '-',
            'location': event.source_name or 'Unknown source',
            'event_type': event_label,
            'confidence': f'{event.confidence * 100:.1f}%' if event.confidence is not None else '-',
            'severity': event_severity,
        })

    recognized_count = sum(
        1 for event in report_events if 'recogn' in event['event_type'].lower()
    )
    unrecognized_count = sum(
        1 for event in report_events
        if any(keyword in event['event_type'].lower() for keyword in ('unknown', 'critical', 'alert'))
    )

    return {
        'selected_type': report_type,
        'timeframe': timeframe,
        'zone': zone,
        'severity': severity,
        'report_title': REPORT_TITLES.get(report_type, REPORT_TITLES['village']),
        'total_triggers': len(report_events),
        'verified_residents': recognized_count,
        'unrecognized_alerts': unrecognized_count,
        'uptime_rate': 'N/A',
        'incident_logs': report_events,
        'date_generated': now.strftime('%Y-%m-%d'),
        'doc_id': f'EBV-{now.strftime("%Y%m%d")}-{len(report_events):04d}',
    }