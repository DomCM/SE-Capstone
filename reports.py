import csv
import io
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


REPORT_TITLES = {
    'village': 'Village-Wide Security Summary Report',
    'incident': 'Security Incident Report',
    'frequency': 'Alert Frequency Analytics Report',
    'surveillance': 'Surveillance System Health Report',
}

REPORT_TIMEZONE = ZoneInfo('Asia/Manila')


def _as_philippine_datetime(value):
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(REPORT_TIMEZONE)


def _philippine_datetime_as_utc_naive(value):
    return value.replace(tzinfo=REPORT_TIMEZONE).astimezone(timezone.utc).replace(tzinfo=None)


def _as_utc_naive(value):
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _severity(event_type):
    normalized = (event_type or '').lower()
    if any(keyword in normalized for keyword in ('critical', 'unknown', 'alert', 'error')):
        return 'High'
    if any(keyword in normalized for keyword in ('motion', 'object', 'detection')):
        return 'Warning'
    return 'Normal'


def _event_severity(event):
    value = getattr(event, 'severity', None)
    if value:
        normalized = str(value).lower()
        if normalized in {'critical', 'high'}:
            return 'High'
        if normalized in {'warning', 'medium'}:
            return 'Warning'
        if normalized in {'low', 'normal'}:
            return 'Normal'
    return _severity(getattr(event, 'event_type', None))


def _metadata_zone(event):
    payload = _event_metadata(event)
    if payload:
        zone_value = payload.get('camera_zone') or payload.get('zone')
        if zone_value:
            return zone_value
    return None


def _event_metadata(event):
    event_metadata = getattr(event, 'event_metadata', None) or ''
    if not event_metadata:
        return {}
    try:
        import json
        payload = json.loads(event_metadata)
        if isinstance(payload, dict):
            return payload
    except (TypeError, ValueError):
        pass
    return {}


def _is_unidentified_or_unusable_person_event(event):
    event_type = str(event.get('raw_event_type') or event.get('event_type') or '').lower()
    is_person_detection = (
        'person' in event_type
        or 'face' in event_type
        or str(event.get('detector') or '').lower() == 'face'
        or str(event.get('object_class') or '').lower() == 'person'
    )
    has_identity_issue = any(
        marker in event_type
        for marker in (
            'unknown',
            'unidentified',
            'unidentifiable',
            'not identifiable',
            'face not visible',
            'face unavailable',
            'face not usable',
            'no usable face',
        )
    )
    return is_person_detection and has_identity_issue


def _resolved_zone(event):
    camera_zone = (getattr(event.camera, 'zone', None) if getattr(event, 'camera', None) else None) or ''
    metadata_zone = _metadata_zone(event) or ''
    camera_name = (getattr(event.camera, 'name', None) if getattr(event, 'camera', None) else None) or ''
    source_name = (getattr(event, 'source_name', None) or '')

    if camera_zone:
        return camera_zone
    if metadata_zone and metadata_zone.lower() not in {camera_name.lower(), source_name.lower()}:
        return metadata_zone
    return 'Main Area'


def _matches_zone(source_name, zone, event=None):
    if not zone or zone == 'all':
        return True

    target_zone = zone.strip().lower()
    candidate_source = (source_name or '').lower()
    metadata_zone = (_metadata_zone(event) or '').lower() if event else ''
    
    event_camera_zone = ''
    if event and getattr(event, 'camera', None):
        event_camera_zone = (getattr(event.camera, 'zone', '') or '').lower()

    combined_text = f"{candidate_source} {metadata_zone} {event_camera_zone}".strip()
    
    zone_aliases = {
        'gate1': ('gate 1', 'gate1', 'cam 1', 'entrance'),
        'north': ('north', 'cam 2', 'perimeter'),
        'clubhouse': ('clubhouse', 'amenities', 'cam 3'),
    }
    aliases = zone_aliases.get(target_zone, (target_zone,))
    return any(alias in combined_text for alias in aliases)


def _matches_severity(event_severity, severity):
    if not severity or severity == 'all':
        return True
    if severity == 'critical':
        return event_severity == 'High'
    if severity == 'normal':
        return event_severity == 'Normal'
    return True


def compute_patrol_priority_summary(events):
    """Score camera activity, limiting repeated public-camera identity events."""
    zone_stats = {}
    public_person_event_scores = {}

    for event in events or []:
        zone = (event.get('zone') or event.get('location') or 'Unassigned').strip() or 'Unassigned'
        camera = event.get('camera_name') or 'Unknown camera'
        camera_key = event.get('camera_id') or (zone, camera)
        key = (zone, camera)
        stats = zone_stats.setdefault(key, {
            'zone': zone,
            'camera_name': camera,
            'total_triggers': 0,
            'high_severity': 0,
            'unknown_faces': 0,
            'crowd_events': 0,
            'vehicle_crossings': 0,
            'risk_score': 0,
            'event_types': {},
        })

        stats['total_triggers'] += 1
        if event.get('severity') == 'High':
            stats['high_severity'] += 1

        event_type = str(event.get('event_type') or '').lower()
        is_identity_issue = _is_unidentified_or_unusable_person_event(event)
        if is_identity_issue:
            stats['unknown_faces'] += 1
        if 'crowd' in event_type:
            stats['crowd_events'] += 1
        if 'vehicle' in event_type:
            stats['vehicle_crossings'] += 1

        is_limited_public_person_event = event.get('camera_is_public', False) and is_identity_issue
        if is_limited_public_person_event:
            previous_score = public_person_event_scores.get(camera_key, 0)
            event_risk_score = 1 if previous_score < 5 else 0
            public_person_event_scores[camera_key] = min(5, previous_score + 1)
        elif 'behavior' in event_type:
            behavior_score = event.get('behavior_score') or 0
            event_risk_score = 12 if behavior_score >= 0.65 else 7 if behavior_score >= 0.4 else 2
        elif 'unattended' in event_type:
            event_risk_score = 6
        elif 'critical' in event_type or 'alert' in event_type:
            event_risk_score = 12
        elif 'unknown' in event_type or 'unidentified' in event_type or 'unidentifiable' in event_type:
            event_risk_score = 9
        elif 'crowd' in event_type:
            event_risk_score = 6
        elif 'vehicle' in event_type or 'crossing' in event_type:
            event_risk_score = 5
        elif 'motion' in event_type:
            event_risk_score = 2
        elif 'recogn' in event_type:
            event_risk_score = 1
        else:
            event_risk_score = 1

        if not is_limited_public_person_event:
            event_risk_score += 2 if event.get('severity') == 'High' else 0
            if 'unknown' in event_type or 'unidentified' in event_type or 'unidentifiable' in event_type:
                event_risk_score += 3
            if 'crowd' in event_type:
                event_risk_score += 2
            if 'vehicle' in event_type:
                event_risk_score += 2
        stats['risk_score'] += event_risk_score

        event_label = event.get('event_type') or 'Other'
        stats['event_types'][event_label] = stats['event_types'].get(event_label, 0) + 1

    summary = []
    for entry in zone_stats.values():
        risk_score = entry['risk_score']
        if risk_score >= 24:
            risk_level = 'High'
            patrol_recommendation = 'Escalate physical patrols and increase monitoring in this zone.'
        elif risk_score >= 10:
            risk_level = 'Medium'
            patrol_recommendation = 'Maintain active patrol checks and review the area during peak hours.'
        else:
            risk_level = 'Low'
            patrol_recommendation = 'Routine patrol cadence is adequate for this location.'

        strongest_event = max(entry['event_types'].items(), key=lambda item: item[1])[0] if entry['event_types'] else 'Routine activity'
        summary.append({
            'zone': entry['zone'],
            'camera_name': entry['camera_name'],
            'total_triggers': entry['total_triggers'],
            'high_severity': entry['high_severity'],
            'unknown_faces': entry['unknown_faces'],
            'risk_score': risk_score,
            'risk_level': risk_level,
            'anomaly_flag': 'Yes' if risk_level != 'Low' else 'No',
            'patrol_recommendation': patrol_recommendation,
            'top_event': strongest_event,
        })

    summary.sort(key=lambda item: item['risk_score'], reverse=True)
    return summary


def build_report(
    detection_event_model,
    report_type='incident',
    timeframe='7d',
    zone='all',
    severity='all',
    start_date=None,
    end_date=None,
    now=None,
    camera_model=None,
):
    """Build the admin report context from detection events."""
    now = _as_utc_naive(now or datetime.utcnow())
    report_type = report_type if report_type in REPORT_TITLES else 'incident'
    timeframe = timeframe if timeframe in {'24h', '7d', '30d', 'custom'} else '7d'
    severity = severity if severity in {'all', 'critical', 'normal'} else 'all'

    # Extract dynamic camera zones from database
    available_zones = []
    if camera_model:
        cameras = camera_model.query.filter(camera_model.is_public.is_(True)).all()
        for cam in cameras:
            cam_zone = (getattr(cam, 'zone', None) or '').strip()
            if cam_zone and cam_zone not in available_zones:
                available_zones.append(cam_zone)

    if not available_zones:
        available_zones = ['Main Entrance', 'North Perimeter', 'Clubhouse & Amenities']

    since = None
    until = None

    if timeframe == 'custom' and start_date and end_date:
        try:
            local_start = datetime.strptime(start_date, '%Y-%m-%d')
            local_end = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
            since = _philippine_datetime_as_utc_naive(local_start)
            until = _philippine_datetime_as_utc_naive(local_end)
        except (ValueError, TypeError):
            timeframe = '7d'

    if not since:
        timeframe_days = {'24h': 1, '7d': 7, '30d': 30}.get(timeframe, 7)
        since = now - timedelta(days=timeframe_days)

    filters = [
        detection_event_model.timestamp >= since,
    ]
    if until:
        filters.append(detection_event_model.timestamp < until)

    query = detection_event_model.query.filter(*filters)
    if camera_model:
        query = query.join(
            camera_model,
            detection_event_model.camera_id == camera_model.id,
        ).filter(camera_model.is_public.is_(True))
    events = query.order_by(detection_event_model.timestamp.desc()).all()

    report_events = []
    for event in events:
        camera = getattr(event, 'camera', None)
        if camera is None or not getattr(camera, 'is_public', False):
            continue

        event_severity = _event_severity(event)
        if not _matches_zone(event.source_name, zone, event) or not _matches_severity(event_severity, severity):
            continue
        event_label = event.event_type or 'Unspecified event'
        if event_label.upper() == 'CRITICAL ALERT' and event.description:
            event_label = f'{event_label}: {event.description.removeprefix("Detected: ").strip()}'
        
        event_zone = _resolved_zone(event)
        cam_name = event.source_name or 'Unknown camera'
        metadata = _event_metadata(event)
        camera_public_value = bool(camera.is_public)
        behavior_score = metadata.get('behavior_score')
        try:
            behavior_score = float(behavior_score) if behavior_score is not None else None
        except (TypeError, ValueError):
            behavior_score = None
        behavior_flags = metadata.get('behavior_flags')
        if not isinstance(behavior_flags, list):
            behavior_flags = []
        unattended_seconds = metadata.get('unattended_seconds')
        nearest_person_distance = metadata.get('nearest_person_distance_pixels')

        report_events.append({
            'event_id': event.id,
            'camera_id': event.camera_id,
            'camera_is_public': camera_public_value,
            'snapshot_available': bool(getattr(event, 'snapshot_path', None)),
            'timestamp': _as_philippine_datetime(event.timestamp).strftime('%Y-%m-%d %I:%M:%S %p %z') if event.timestamp else '-',
            'raw_timestamp': event.timestamp,
            'location': event_zone,
            'zone': event_zone,
            'camera_name': cam_name,
            'event_type': event_label,
            'raw_event_type': event.event_type,
            'detector': event.detector,
            'object_class': metadata.get('class'),
            'confidence': f'{event.confidence * 100:.1f}%' if event.confidence is not None else '-',
            'severity': event_severity,
            'behavior_score': behavior_score,
            'behavior_flags': [str(flag) for flag in behavior_flags],
            'track_id': metadata.get('track_id'),
            'unattended_seconds': unattended_seconds,
            'nearest_person_distance_pixels': nearest_person_distance,
        })

    behavior_events = [
        event for event in report_events
        if event.get('behavior_score') is not None
    ]
    behavior_flag_counts = {}
    for event in behavior_events:
        for flag in event.get('behavior_flags', []):
            behavior_flag_counts[flag] = behavior_flag_counts.get(flag, 0) + 1
    behavior_scores = [event['behavior_score'] for event in behavior_events]
    behavior_summary = {
        'alert_count': len(behavior_events),
        'average_score': round(sum(behavior_scores) / len(behavior_scores), 3) if behavior_scores else 0,
        'peak_score': max(behavior_scores, default=0),
        'flag_counts': [
            {'label': flag, 'value': count}
            for flag, count in sorted(
                behavior_flag_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
    }

    # Incident KPIs use the same timeframe, zone, and severity filters as the report.
    high_severity_incidents = sum(
        1 for event in report_events if event['severity'] == 'High'
    )
    zone_incident_counts = {}
    for event in report_events:
        zone = event['zone']
        zone_incident_counts[zone] = zone_incident_counts.get(zone, 0) + 1
    affected_zone_count = len(zone_incident_counts)
    hotspot_zone, hotspot_incident_count = (
        min(
            zone_incident_counts.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )
        if zone_incident_counts else ('No incidents', 0)
    )
    snapshot_evidence_count = sum(
        1 for event in report_events if event['snapshot_available']
    )
    recognized_count = sum(
        1 for event in report_events if 'recogn' in event['event_type'].lower()
    )
    unrecognized_count = sum(
        1 for event in report_events
        if _is_unidentified_or_unusable_person_event(event)
        or any(keyword in event['event_type'].lower() for keyword in ('unknown', 'critical', 'alert'))
    )

    # Filtered Incident Logs for Incident Report
    if report_type == 'incident':
        filtered_events = [
            e for e in report_events
            if e['severity'] in {'High', 'Warning'} or 'unknown' in e['event_type'].lower()
        ]
    else:
        filtered_events = report_events

    # Frequency Analytics Breakdown by Zone / Camera Name
    patrol_priority_summary = []
    frequency_summary = []
    if report_type == 'frequency':
        location_data = {}
        for event in report_events:
            camera_label = event.get('camera_name') or event.get('location') or 'Unknown camera'
            key = (event['zone'], camera_label)
            if key not in location_data:
                location_data[key] = {
                    'zone': event['zone'],
                    'camera_name': camera_label,
                    'total_triggers': 0,
                    'high_severity': 0,
                    'hours': {},
                    'types': {},
                    'behavior_scores': [],
                    'behavior_flags': {},
                }
            location_data[key]['total_triggers'] += 1
            if event['severity'] == 'High':
                location_data[key]['high_severity'] += 1
            
            ts = event.get('raw_timestamp')
            if ts:
                hour_str = _as_philippine_datetime(ts).strftime('%I:00 %p')
                location_data[key]['hours'][hour_str] = location_data[key]['hours'].get(hour_str, 0) + 1
            
            ev_type = event['event_type']
            location_data[key]['types'][ev_type] = location_data[key]['types'].get(ev_type, 0) + 1
            if event.get('behavior_score') is not None:
                location_data[key]['behavior_scores'].append(event['behavior_score'])
                for flag in event.get('behavior_flags', []):
                    counts = location_data[key]['behavior_flags']
                    counts[flag] = counts.get(flag, 0) + 1

        for (z, c), data in location_data.items():
            peak_h = max(data['hours'], key=data['hours'].get) if data['hours'] else 'N/A'
            top_t = max(data['types'], key=data['types'].get) if data['types'] else 'N/A'
            frequency_summary.append({
                'zone': z,
                'camera_name': c,
                'location': f"{z} ({c})",
                'total_triggers': data['total_triggers'],
                'high_severity': data['high_severity'],
                'peak_hour': peak_h,
                'top_event': top_t,
                'behavior_alerts': len(data['behavior_scores']),
                'peak_behavior_score': max(data['behavior_scores'], default=0),
                'top_behavior_flag': (
                    max(data['behavior_flags'], key=data['behavior_flags'].get)
                    if data['behavior_flags'] else 'N/A'
                ),
            })
        frequency_summary.sort(key=lambda x: x['total_triggers'], reverse=True)

        patrol_priority_summary = compute_patrol_priority_summary(report_events)
        priority_lookup = {(item['zone'], item['camera_name']): item for item in patrol_priority_summary}
        for row in frequency_summary:
            priority = priority_lookup.get((row['zone'], row['camera_name']), {})
            row['risk_score'] = priority.get('risk_score', 0)
            row['risk_level'] = priority.get('risk_level', 'Low')
            row['patrol_recommendation'] = priority.get('patrol_recommendation', 'Routine patrol cadence is adequate for this location.')
            row['anomaly_flag'] = priority.get('anomaly_flag', 'No')

        zone_totals = {}
        for item in patrol_priority_summary:
            zone_name = item['zone']
            zone_totals[zone_name] = zone_totals.get(zone_name, 0) + item['risk_score']

        zone_risk_chart = [
            {'label': zone, 'value': score}
            for zone, score in sorted(zone_totals.items(), key=lambda x: x[1], reverse=True)[:6]
        ]

        hourly_activity = {}
        for event in report_events:
            ts = event.get('raw_timestamp')
            if ts is None:
                continue
            hour_key = _as_philippine_datetime(ts).strftime('%H:00')
            hourly_activity[hour_key] = hourly_activity.get(hour_key, 0) + 1
        hourly_activity_chart = [
            {'label': hour, 'value': count}
            for hour, count in sorted(hourly_activity.items())
        ]

        severity_counts = {'High': 0, 'Warning': 0, 'Normal': 0}
        for event in report_events:
            level = event.get('severity') or 'Normal'
            if level in severity_counts:
                severity_counts[level] += 1
        severity_chart = [
            {'label': label, 'value': value}
            for label, value in severity_counts.items()
            if value > 0
        ]
    else:
        zone_risk_chart = []
        hourly_activity_chart = []
        severity_chart = []

    # Surveillance System Health Breakdown
    surveillance_summary = []
    uptime_rate = '100.0%'
    offline_event_count = 0

    if camera_model:
        cameras = camera_model.query.filter(camera_model.is_public.is_(True)).all()
        for cam in cameras:
            cam_zone = getattr(cam, 'zone', 'Main Area') or 'Main Area'
            cam_events = [
                e for e in report_events
                if (e.get('camera_name') or '').lower() == cam.name.lower()
                or cam_zone.lower() in (e.get('zone') or '').lower()
            ]
            disconnects = sum(1 for e in cam_events if 'offline' in e['event_type'].lower() or 'disconnect' in e['event_type'].lower())
            offline_event_count += disconnects
            last_ts = cam_events[0]['timestamp'] if cam_events else 'No recent logs'
            status = 'Operational' if disconnects == 0 else f'{disconnects} Disconnect Alert(s)'
            surveillance_summary.append({
                'zone': cam_zone,
                'camera_name': cam.name,
                'status': status,
                'disconnect_events': disconnects,
                'last_activity': last_ts,
            })
        if cameras and offline_event_count > 0:
            total_evals = max(len(report_events), 1)
            calculated = max(0.0, 100.0 - (offline_event_count / total_evals * 100))
            uptime_rate = f'{calculated:.1f}%'
    else:
        uptime_rate = '99.8%' if len(report_events) > 0 else '100.0%'

    return {
        'selected_type': report_type,
        'timeframe': timeframe,
        'zone': zone,
        'severity': severity,
        'start_date': start_date or '',
        'end_date': end_date or '',
        'available_zones': available_zones,
        'report_title': REPORT_TITLES.get(report_type, REPORT_TITLES['village']),
        'total_triggers': len(report_events),
        'verified_residents': recognized_count,
        'unrecognized_alerts': unrecognized_count,
        'uptime_rate': uptime_rate,
        'high_severity_incidents': high_severity_incidents,
        'affected_zone_count': affected_zone_count,
        'incident_hotspot_zone': hotspot_zone,
        'incident_hotspot_count': hotspot_incident_count,
        'snapshot_evidence_count': snapshot_evidence_count,
        'snapshot_evidence_total': len(report_events),
        'incident_logs': filtered_events,
        'frequency_summary': frequency_summary,
        'patrol_priority_summary': patrol_priority_summary,
        'behavior_summary': behavior_summary,
        'zone_risk_chart': zone_risk_chart,
        'hourly_activity_chart': hourly_activity_chart,
        'severity_chart': severity_chart,
        'surveillance_summary': surveillance_summary,
        'date_generated': _as_philippine_datetime(now).strftime('%Y-%m-%d'),
        'doc_id': f'EBV-{_as_philippine_datetime(now).strftime("%Y%m%d")}-{len(report_events):04d}',
    }


def generate_report_csv(report_data):
    """Generate a clean, standardized CSV representation of the report data."""
    output = io.StringIO()
    writer = csv.writer(output)

    report_type = report_data.get('selected_type', 'village')

    if report_type == 'frequency':
        writer.writerow(['Assigned Zone', 'Camera Name', 'Total Triggers', 'High Severity Alerts', 'Peak Hour', 'Primary Trigger Type', 'Behavior Alerts', 'Peak Behavior Score', 'Top Behavior Flag', 'Risk Score', 'Patrol Priority', 'Patrol Recommendation'])
        for row in report_data.get('frequency_summary', []):
            writer.writerow([
                row.get('zone', '-'),
                row.get('camera_name', '-'),
                row.get('total_triggers', 0),
                row.get('high_severity', 0),
                row.get('peak_hour', '-'),
                row.get('top_event', '-'),
                row.get('behavior_alerts', 0),
                row.get('peak_behavior_score', 0),
                row.get('top_behavior_flag', 'N/A'),
                row.get('risk_score', 0),
                row.get('risk_level', 'Low'),
                row.get('patrol_recommendation', '-'),
            ])
    elif report_type == 'surveillance':
        writer.writerow(['Assigned Zone', 'Camera Name', 'Operational Status', 'Disconnect Events', 'Last Recorded Activity'])
        for row in report_data.get('surveillance_summary', []):
            writer.writerow([
                row.get('zone', '-'),
                row.get('camera_name', '-'),
                row.get('status', '-'),
                row.get('disconnect_events', 0),
                row.get('last_activity', '-'),
            ])
    else:
        writer.writerow(['Timestamp', 'Assigned Zone', 'Camera Name', 'Event / Trigger Type', 'Confidence', 'Severity Status', 'Behavior Score', 'Behavior Flags', 'Track ID', 'Unattended Seconds', 'Nearest Person Distance (px)'])
        for row in report_data.get('incident_logs', []):
            writer.writerow([
                row.get('timestamp', '-'),
                row.get('location', '-'),
                row.get('camera_name', row.get('location', '-')),
                row.get('event_type', '-'),
                row.get('confidence', '-'),
                row.get('severity', '-'),
                row.get('behavior_score', ''),
                '; '.join(row.get('behavior_flags', [])),
                row.get('track_id', ''),
                row.get('unattended_seconds', ''),
                row.get('nearest_person_distance_pixels', ''),
            ])

    return output.getvalue()