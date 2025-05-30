from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import pandas as pd
import sys
sys.path.append('../../')
from utils import read_stub, save_stub


class BallTracker:
    def __init__(self, model_path, max_age=15, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.tracker = DeepSort(max_age=max_age)
        self.conf_threshold = conf_threshold

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=self.conf_threshold)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None and len(tracks) == len(frames):
            return tracks

        detections = self.detect_frames(frames)
        print(detections)
        result_tracks = []

        for i, detection in enumerate(detections):
            frame = frames[i]
            frame_tracks = {}

            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            ball_cls_id = cls_names_inv.get('Ball')

            det_inputs = []
            for box, conf, cls in zip(
                detection.boxes.xyxy.cpu().numpy(),
                detection.boxes.conf.cpu().numpy(),
                detection.boxes.cls.cpu().numpy()
            ):
                if cls == ball_cls_id and conf >= self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    w, h = x2 - x1, y2 - y1
                    det_inputs.append([[x1, y1, w, h], float(conf), 'ball'])

            updated_tracks = self.tracker.update_tracks(det_inputs, frame=frame)

            for track in updated_tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                x, y, w, h = map(int, track.to_tlwh())
                frame_tracks[tid] = {"bbox": [x, y, x + w, y + h]}

            result_tracks.append(frame_tracks)

        save_stub(stub_path, result_tracks)
        return result_tracks

    def remove_wrong_detections(self, ball_positions, max_distance=25):
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current = list(ball_positions[i].values())
            if not current:
                continue
            current_box = current[0]['bbox']

            if last_good_frame_index == -1:
                last_good_frame_index = i
                continue

            last_box = list(ball_positions[last_good_frame_index].values())[0]['bbox']
            dist = np.linalg.norm(np.array(current_box[:2]) - np.array(last_box[:2]))
            if dist > max_distance * (i - last_good_frame_index):
                ball_positions[i] = {}
            else:
                last_good_frame_index = i

        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        bboxes = []
        for d in ball_positions:
            if d:
                bbox = list(d.values())[0]['bbox']
            else:
                bbox = [np.nan] * 4
            bboxes.append(bbox)

        df = pd.DataFrame(bboxes, columns=['x1', 'y1', 'x2', 'y2']).interpolate().bfill()
        return [{1: {"bbox": row.tolist()}} for _, row in df.iterrows()]
