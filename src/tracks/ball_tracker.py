

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys
sys.path.append('../../')  

from src.utils import read_stub, save_stub


class BallTracker:
    """
    A class for ball detection and tracking using YOLOv8 and Deep SORT.
    """

    def __init__(
        self,
        model_path: str = "models/ball_detector_model.pt",  
        max_age: int = 30,
        conf_threshold: float = 0.5
    ):
        """
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
        Run YOLO detection (par frame) sur une liste de frames.
        Returns une liste de résultats de détection.
        """
        detections = []
        for frame in frames:
            result = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
            detections.append(result[0])
        return detections

    def track_ball(self, frames, use_cache=False, cache_path=None):
        """
        Track the ball over a sequence of frames, with optional caching.

        Args:
            frames (List[np.ndarray]): Video frames.
            use_cache (bool): Si True, tente de charger depuis le cache.
            cache_path (str): Chemin vers le fichier de cache.

        Returns:
            List[Dict[int, Dict]]: Pour chaque frame, un dict mapping track_id -> {"bbox": [x, y, w, h]}.
        """
        # 1) Tentative de chargement du cache
        cached = read_stub(use_cache, cache_path)
        if cached is not None and len(cached) == len(frames):
            return cached

        # 2) Détection
        detections = self.process_batches(frames)
        tracks_per_frame = []

        for i, det in enumerate(detections):
            # Trouver l'ID de la classe "Ball"
            inv_names = {v: k for k, v in det.names.items()}
            ball_cls_id = inv_names.get('Ball', None)

            # Préparer les entrées pour Deep SORT
            det_inputs = []
            for box, score, cls in zip(
                det.boxes.xyxy.cpu().numpy(),
                det.boxes.conf.cpu().numpy(),
                det.boxes.cls.cpu().numpy()
            ):
                if cls == ball_cls_id and score >= self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    w, h = x2 - x1, y2 - y1
                    det_inputs.append([[x1, y1, w, h], float(score), 'ball'])

            # Mettre à jour le tracker
            updated = self.tracker.update_tracks(det_inputs, frame=frames[i])
            frame_tracks = {}
            for t in updated:
                if not t.is_confirmed():
                    continue
                tid = t.track_id
                l, t_, w, h = map(int, t.to_tlwh())
                frame_tracks[tid] = {"bbox": [l, t_, w, h]}

            tracks_per_frame.append(frame_tracks)

        save_stub(cache_path, tracks_per_frame)
        return tracks_per_frame
