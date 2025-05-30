from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


import sys
sys.path.append('../../')

from src.utils import read_stub, save_stub


class PlayerTracker:
    """
    A class for player detection and tracking using YOLOv8 and Deep SORT.
    """

    def __init__(self, model_path, max_age=30, conf_threshold=0.5):
        """
        Initialize the YOLOv8 model and Deep SORT tracker.

        Args:
            model_path (str): Path to the YOLO model weights.
            max_age (int): Max number of frames to keep a lost track.
            conf_threshold (float): Confidence threshold for detections.
        """
        self.model = YOLO(model_path)
        self.tracker = DeepSort(max_age=max_age)
        self.conf_threshold = conf_threshold

    def process_batches(self, frames):
        """
        Run YOLO detection in batches on a list of frames.

        Args:
            frames (list): List of frames to process.

        Returns:
            list: YOLO detection results.
        """
        detections = []
        for frame in frames:
            result = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
            detections.append(result[0])
        return detections

    def track_players(self, frames, use_cache=False, cache_path=None):
        """
        Track players across frames and return tracking results.

        Args:
            frames (list): List of video frames.
            use_cache (bool): Whether to read from a cached result.
            cache_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries. Each dict maps track IDs to bounding boxes.
        """
        cached = read_stub(use_cache, cache_path)
        if cached is not None and len(cached) == len(frames):
            return cached

        detections = self.process_batches(frames)
        tracks_per_frame = []

        for i, detection in enumerate(detections):
            tracks_frame = {}
            detections_input = []

            for box, score, cls in zip(detection.boxes.xyxy.cpu().numpy(),
                                       detection.boxes.conf.cpu().numpy(),
                                       detection.boxes.cls.cpu().numpy()):
                if int(cls) == 4 and score >= self.conf_threshold:  # Class 4 = Player
                    x1, y1, x2, y2 = map(int, box)
                    w, h = x2 - x1, y2 - y1
                    bbox = [x1, y1, w, h]
                    detections_input.append([bbox, score, 'player'])

            # Update tracker and get current tracks
            updated_tracks = self.tracker.update_tracks(detections_input, frame=frames[i])

            for t in updated_tracks:
                if not t.is_confirmed():
                    continue
                track_id = t.track_id
                l, t_, w, h = map(int, t.to_tlwh())
                tracks_frame[track_id] = {"bbox": [l, t_, w, h]}

            tracks_per_frame.append(tracks_frame)

        save_stub(cache_path, tracks_per_frame)
        return tracks_per_frame
