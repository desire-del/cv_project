import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from src.tactic_view.homography import Homography

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, "../../"))
from src.utils import get_foot_position, measure_distance

class TacticalViewConverter:
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.width = 300
        self.height = 161

        self.actual_width_in_meters = 28
        self.actual_height_in_meters = 15

        self.key_points = self._generate_key_points()

    def _generate_key_points(self):
        def px(x_m, y_m):
            return (
                int((x_m / self.actual_width_in_meters) * self.width),
                int((y_m / self.actual_height_in_meters) * self.height)
            )

        return [
            # Left edge
            px(0, 0), px(0, 0.91), px(0, 5.18), px(0, 10), px(0, 14.1), px(0, 15),
            # Middle line
            (self.width // 2, self.height), (self.width // 2, 0),
            # Left free throw
            px(5.79, 5.18), px(5.79, 10),
            # Right edge
            px(28, 15), px(28, 14.1), px(28, 10), px(28, 5.18), px(28, 0.91), px(28, 0),
            # Right free throw
            px(28 - 5.79, 5.18), px(28 - 5.79, 10)
        ]

    def validate_keypoints(self, keypoints_list):
        validated_list = deepcopy(keypoints_list)

        for frame_idx, frame_keypoints in enumerate(validated_list):
            if len(frame_keypoints) == 0:
                continue  # aucun keypoint

            frame_keypoints = frame_keypoints[0]  # shape (N, 2)

            detected_indices = [i for i, kp in enumerate(frame_keypoints) if kp[0] > 0 and kp[1] > 0]

            if len(detected_indices) < 3:
                continue

            invalid_keypoints = []

            for i in detected_indices:
                other_indices = [idx for idx in detected_indices if idx != i and idx not in invalid_keypoints]
                if len(other_indices) < 2:
                    continue

                j, k = other_indices[:2]
                d_ij = measure_distance(frame_keypoints[i], frame_keypoints[j])
                d_ik = measure_distance(frame_keypoints[i], frame_keypoints[k])

                t_ij = measure_distance(self.key_points[i], self.key_points[j])
                t_ik = measure_distance(self.key_points[i], self.key_points[k])

                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij / d_ik if d_ik > 0 else float('inf')
                    prop_tactical = t_ij / t_ik if t_ik > 0 else float('inf')

                    error = abs((prop_detected - prop_tactical) / prop_tactical)

                    if error > 0.8:
                        validated_list[frame_idx][0][i] = np.array([0.0, 0.0])
                        invalid_keypoints.append(i)

        return validated_list

    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
        tactical_positions_per_frame = []

        for frame_keypoints, frame_tracks in zip(keypoints_list, player_tracks):
            tactical_positions = {}
            if len(frame_keypoints) == 0:
                tactical_positions_per_frame.append(tactical_positions)
                continue

            frame_keypoints = frame_keypoints[0]  # shape (N, 2)

            valid_indices = [i for i, kp in enumerate(frame_keypoints) if kp[0] > 0 and kp[1] > 0]

            if len(valid_indices) < 4:
                tactical_positions_per_frame.append(tactical_positions)
                continue

            source_points = np.array([frame_keypoints[i] for i in valid_indices], dtype=np.float32)
            target_points = np.array([self.key_points[i] for i in valid_indices], dtype=np.float32)

            try:
                homography = Homography(source_points, target_points)
                for player_id, data in frame_tracks.items():
                    player_position = np.array([get_foot_position(data["bbox"])]).astype(np.float32)
                    tactical_pos = homography.transform_points(player_position)
                    x, y = tactical_pos[0]
                    if 0 <= x <= self.width and 0 <= y <= self.height:
                        tactical_positions[player_id] = [float(x), float(y)]
            except (ValueError, cv2.error):
                pass

            tactical_positions_per_frame.append(tactical_positions)

        return tactical_positions_per_frame
