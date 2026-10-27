import collections
import math

import numpy as np


class SpatialDensityMap:
    def __init__(self, grid_size=(20, 20), decay_factor=0.999):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size, dtype=np.float32)
        self.decay_factor = decay_factor

    def update_and_evaluate(self, point, frame_shape):
        height, width = frame_shape[:2]
        grid_x = int(np.clip(point[0] / max(1, width) * self.grid_size[1], 0, self.grid_size[1] - 1))
        grid_y = int(np.clip(point[1] / max(1, height) * self.grid_size[0], 0, self.grid_size[0] - 1))
        self.grid *= self.decay_factor
        self.grid[grid_y, grid_x] += 1.0

        total_density = float(np.sum(self.grid))
        if total_density < 10.0:
            return 0.0

        cell_density = float(self.grid[grid_y, grid_x] / total_density)
        if cell_density < 0.005:
            return 0.8
        if cell_density < 0.02:
            return 0.4
        return 0.0


class AdvancedTrajectoryTracker:
    def __init__(self, max_history=45, association_distance=160.0):
        self.max_history = max_history
        self.association_distance = association_distance
        self.tracks = {}
        self.first_seen = {}
        self.last_seen = {}
        self.threat_energy = {}
        self.last_energy_time = {}
        self.spatial_density = SpatialDensityMap()
        self._track_boxes = {}
        self._next_track_id = 0

    def associate_detections(self, detections, current_time):
        for record in self._track_boxes.values():
            record.pop('assigned', None)
        candidates = []
        for detection_index, detection in enumerate(detections):
            box = tuple(float(value) for value in detection[:4])
            center = ((box[0] + box[2]) / 2.0, box[3])
            for track_id, record in self._track_boxes.items():
                if current_time - record['last_seen'] > 3.0:
                    continue
                old_center = record['center']
                distance = math.hypot(center[0] - old_center[0], center[1] - old_center[1])
                max_distance = max(
                    self.association_distance,
                    (box[2] - box[0]) * 1.5,
                    (box[3] - box[1]) * 0.75,
                )
                if distance <= max_distance:
                    candidates.append((distance, detection_index, track_id))

        assignments = {}
        used_tracks = set()
        for _, detection_index, track_id in sorted(candidates):
            if detection_index not in assignments and track_id not in used_tracks:
                assignments[detection_index] = track_id
                used_tracks.add(track_id)

        for detection_index, detection in enumerate(detections):
            box = tuple(float(value) for value in detection[:4])
            center = ((box[0] + box[2]) / 2.0, box[3])
            track_id = assignments.get(detection_index)
            if track_id is None:
                self._next_track_id += 1
                track_id = self._next_track_id
                assignments[detection_index] = track_id
            self._track_boxes[track_id] = {
                'bbox': box,
                'center': center,
                'last_seen': current_time,
            }

        for record in self._track_boxes.values():
            record.pop('assigned', None)
        for track_id in assignments.values():
            self._track_boxes[track_id]['assigned'] = True
        self._track_boxes = {
            track_id: record
            for track_id, record in self._track_boxes.items()
            if current_time - record['last_seen'] <= 3.0
        }
        return [assignments[index] for index in range(len(detections))]

    def update(self, track_id, centroid, timestamp):
        if track_id not in self.tracks:
            self.tracks[track_id] = collections.deque(maxlen=self.max_history)
            self.first_seen[track_id] = timestamp
            self.threat_energy[track_id] = 0.0
            self.last_energy_time[track_id] = timestamp

        # Apply exponential moving average smoothing on raw centroid to dampen jitter/frame skip shifts
        points = self.tracks[track_id]
        if len(points) > 0:
            last_x, last_y, _ = points[-1]
            alpha = 0.35  # Smoothing factor
            smoothed_x = alpha * centroid[0] + (1 - alpha) * last_x
            smoothed_y = alpha * centroid[1] + (1 - alpha) * last_y
            centroid = (smoothed_x, smoothed_y)

        self.tracks[track_id].append((centroid[0], centroid[1], timestamp))
        self.last_seen[track_id] = timestamp

    def compute_pacing_entropy(self, track_id):
        points = list(self.tracks.get(track_id, []))
        if len(points) < 6:
            return 0.0

        angles = []
        for previous, current in zip(points, points[1:]):
            dx = current[0] - previous[0]
            dy = current[1] - previous[1]
            dt = max(current[2] - previous[2], 0.001)
            dist = math.hypot(dx, dy)
            # Require minimum displacement relative to time gap to prevent micro-jitter turning angles
            if dist / dt > 15.0:
                angles.append(np.arctan2(dy, dx))
        if len(angles) < 4:
            return 0.0

        differences = np.abs(np.diff(angles))
        differences = np.where(differences > np.pi, 2 * np.pi - differences, differences)
        return float(np.sum(differences > 2 * np.pi / 3) / len(differences))

    def compute_kinematics(self, track_id):
        points = self.tracks.get(track_id, [])
        if len(points) < 3:
            return 0.0, (0.0, 0.0)

        # Compute velocity using timestamps instead of frame counts
        x1, y1, time1 = points[-3]
        x2, y2, time2 = points[-1]
        elapsed = max(time2 - time1, 0.01)
        dx, dy = x2 - x1, y2 - y1
        distance = math.hypot(dx, dy)
        heading = (dx / distance, dy / distance) if distance > 0 else (0.0, 0.0)
        return distance / elapsed, heading

    def evaluate_shadowing(self, track_id, active_centroids):
        if len(self.tracks.get(track_id, [])) < 5 or len(active_centroids) < 2:
            return 0.0
        current_point = np.asarray(self.tracks[track_id][-1][:2])
        speed, heading = self.compute_kinematics(track_id)
        if speed < 15.0:
            return 0.0

        for other_id, other_point in active_centroids.items():
            if other_id == track_id:
                continue
            distance = np.linalg.norm(current_point - np.asarray(other_point))
            if 30.0 < distance < 110.0:
                other_speed, other_heading = self.compute_kinematics(other_id)
                alignment = heading[0] * other_heading[0] + heading[1] * other_heading[1]
                if alignment > 0.75 and other_speed > 15.0:
                    return 0.75
        return 0.0

    def update_threat_energy(self, track_id, instantaneous_score, current_time, decay_lambda=0.15):
        if track_id not in self.threat_energy:
            self.threat_energy[track_id] = instantaneous_score
            self.last_energy_time[track_id] = current_time
            return instantaneous_score

        elapsed = max(0.0, current_time - self.last_energy_time[track_id])
        self.last_energy_time[track_id] = current_time
        decayed_score = self.threat_energy[track_id] * np.exp(-decay_lambda * elapsed)
        smoothed_score = min(1.0, float(decayed_score) + instantaneous_score * 0.4)
        self.threat_energy[track_id] = smoothed_score
        return smoothed_score

    def get_dwell_time(self, track_id, current_time):
        return max(0.0, current_time - self.first_seen.get(track_id, current_time))

    def purge_stale(self, current_time, max_age=3.0):
        stale_ids = [
            track_id for track_id, last_seen in self.last_seen.items()
            if current_time - last_seen > max_age
        ]
        for track_id in stale_ids:
            self.tracks.pop(track_id, None)
            self.first_seen.pop(track_id, None)
            self.last_seen.pop(track_id, None)
            self.threat_energy.pop(track_id, None)
            self.last_energy_time.pop(track_id, None)
            self._track_boxes.pop(track_id, None)


class BehaviorEvaluator:
    def __init__(self, dwell_threshold=10.0, speed_threshold=320.0):
        self.dwell_threshold = dwell_threshold
        self.speed_threshold = speed_threshold

    def evaluate_pose_anomalies(self, keypoints, bbox):
        if keypoints is None or len(keypoints) < 17:
            return 0.0, []

        nose = keypoints[0]
        left_shoulder, right_shoulder = keypoints[5], keypoints[6]
        left_hip, right_hip = keypoints[11], keypoints[12]
        left_knee, right_knee = keypoints[13], keypoints[14]
        left_ankle, right_ankle = keypoints[15], keypoints[16]
        left_wrist, right_wrist = keypoints[9], keypoints[10]
        x1, y1, x2, y2 = bbox
        box_height = max(1, y2 - y1)
        box_width = max(1, x2 - x1)
        flags = []
        pose_score = 0.0

        # Check torso & leg keypoints confidence
        upper_valid = left_shoulder[2] > 0.45 and right_shoulder[2] > 0.45
        hip_valid = left_hip[2] > 0.45 and right_hip[2] > 0.45
        lower_valid = (left_knee[2] > 0.4 or right_knee[2] > 0.4) or (left_ankle[2] > 0.4 or right_ankle[2] > 0.4)

        if upper_valid and hip_valid:
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
            hip_y = (left_hip[1] + right_hip[1]) / 2.0
            torso_height = abs(hip_y - shoulder_y)
            
            # Additional validation: Hips must actually be below shoulders
            if hip_y > shoulder_y:
                # Require lower body keypoints to confirm crouching vs missing/misaligned upper keypoints
                if lower_valid:
                    lowest_leg_y = max(
                        left_knee[1] if left_knee[2] > 0.4 else 0,
                        right_knee[1] if right_knee[2] > 0.4 else 0,
                        left_ankle[1] if left_ankle[2] > 0.4 else 0,
                        right_ankle[1] if right_ankle[2] > 0.4 else 0,
                    )
                    leg_length = max(1, lowest_leg_y - hip_y)
                    torso_to_leg_ratio = torso_height / leg_length
                    
                    # Crouching typically compresses leg length relative to bounding box height and lowers hips relative to box top
                    hip_rel_pos = (hip_y - y1) / box_height
                    if torso_to_leg_ratio < 0.35 and hip_rel_pos < 0.55 and (box_height / box_width < 1.0):
                        pose_score += 0.5
                        flags.append('CROUCHING_OR_HIDING')

        if all(point[2] > 0.4 for point in (left_wrist, right_wrist, nose)):
            if (left_wrist[1] + right_wrist[1]) / 2.0 < nose[1]:
                pose_score += 0.3
                flags.append('HANDS_RAISED')
        return pose_score, flags

    def evaluate(
        self, dwell_time, speed, pacing_entropy, is_in_zone,
        pose_score=0.0, shadowing_score=0.0, spatial_anomaly=0.0,
    ):
        dwell_score = min(1.0, max(0.0, (dwell_time - 3.0) / self.dwell_threshold))
        kinematic_score = min(1.0, speed / self.speed_threshold) if speed > self.speed_threshold * 0.5 else 0.0
        pacing_score = min(1.0, pacing_entropy * 2.0) if pacing_entropy > 0.3 else 0.0
        zone_score = 0.85 if is_in_zone else 0.0
        score = float(np.clip(
            (0.20 * dwell_score)
            + (0.20 * zone_score)
            + (0.10 * kinematic_score)
            + (0.15 * pacing_score)
            + (0.15 * pose_score)
            + (0.10 * shadowing_score)
            + (0.10 * spatial_anomaly),
            0.0,
            1.0,
        ))

        flags = []
        if dwell_score > 0.6:
            flags.append('LOITERING_BEHAVIOR')
        if zone_score > 0:
            flags.append('ZONE_INTRUSION')
        if kinematic_score > 0.7:
            flags.append('ERRATIC_MOTION')
        if pacing_score > 0.5:
            flags.append('PACING_BEHAVIOR')
        if shadowing_score > 0.5:
            flags.append('TAILGATING_SHADOWING')
        if spatial_anomaly > 0.5:
            flags.append('ABNORMAL_GRID_OCCUPANCY')
        return score, flags


class UnattendedObjectTracker:
    def __init__(self, max_distance=130.0, unattended_seconds=10.0):
        self.max_distance = max_distance
        self.unattended_seconds = unattended_seconds
        self._objects = {}
        self._next_track_id = 0

    def evaluate_unattended_objects(self, object_boxes, person_centroids, current_time):
        alerts = []
        assigned_tracks = set()
        for item in object_boxes:
            box = tuple(int(value) for value in item['box'])
            object_center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            closest_person = min(
                (
                    math.hypot(object_center[0] - point[0], object_center[1] - point[1])
                    for point in person_centroids.values()
                ),
                default=float('inf'),
            )
            previous = min(
                (
                    (math.hypot(object_center[0] - record['center'][0], object_center[1] - record['center'][1]), track_id)
                    for track_id, record in self._objects.items()
                    if track_id not in assigned_tracks
                    and current_time - record['last_seen'] <= 3.0
                    and record['label'] == item['label']
                ),
                default=(float('inf'), None),
            )
            if previous[0] <= self.max_distance:
                track_id = previous[1]
            else:
                self._next_track_id += 1
                track_id = self._next_track_id
                self._objects[track_id] = {
                    'center': object_center,
                    'label': item['label'],
                    'unattended_since': None,
                }
            record = self._objects[track_id]
            record['center'] = object_center
            record['last_seen'] = current_time
            assigned_tracks.add(track_id)

            if closest_person > self.max_distance:
                if record['unattended_since'] is None:
                    record['unattended_since'] = current_time
            else:
                record['unattended_since'] = None

            unattended_for = (
                current_time - record['unattended_since']
                if record['unattended_since'] is not None else 0.0
            )
            if unattended_for >= self.unattended_seconds:
                alerts.append({
                    'label': item['label'],
                    'box': box,
                    'distance': round(closest_person, 1) if math.isfinite(closest_person) else None,
                    'track_id': track_id,
                    'unattended_seconds': round(unattended_for, 1),
                })
        self._objects = {
            track_id: record
            for track_id, record in self._objects.items()
            if current_time - record['last_seen'] <= 3.0
        }
        return alerts