import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class PlayerTracker:
    def __init__(self, yolo_model_path='yolov8n.pt', max_age=30, conf_threshold=0.3):
        """
        Initialize the player tracker.
        
        Args:
            yolo_model_path (str): Path or name of the YOLOv8 model.
            max_age (int): Max frames to keep lost tracks in Deep SORT.
            conf_threshold (float): Confidence threshold for YOLO detections.
        """
        self.model = YOLO(yolo_model_path)
        self.tracker = DeepSort(max_age=max_age)
        self.conf_threshold = conf_threshold

    def update(self, frame):
        """
        Process a frame: detect players and update tracker.
        
        Args:
            frame (np.array): The current video frame (BGR).
        
        Returns:
            annotated_frame (np.array): Frame with drawn bounding boxes and IDs.
            tracks_info (list): List of dicts with keys: track_id, bbox (x,y,w,h).
        """
        results = self.model(frame)[0]

        detections = []
        for box, score, cls_id in zip(results.boxes.xyxy.cpu().numpy(),
                                      results.boxes.conf.cpu().numpy(),
                                      results.boxes.cls.cpu().numpy()):
            if int(cls_id) == 4 and score > self.conf_threshold:  # Player class
                x1, y1, x2, y2 = box.astype(int)
                detections.append(([x1, y1, x2 - x1, y2 - y1], score, 'Player'))

        tracks = self.tracker.update_tracks(detections, frame=frame)

        tracks_info = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = [int(i) for i in track.to_tlwh()]
            tracks_info.append({'track_id': track_id, 'bbox': (l, t, w, h)})

            # Draw bounding box + ID on frame
            cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame, tracks_info


