from src.tracks.player_tracker import PlayerTracker
import cv2



if __name__ == "__main__":

    video_path = "data/videos/video_1.mp4"
    cap = cv2.VideoCapture(video_path)
    tracker = PlayerTracker(yolo_model_path="models/players_detection_model.pt", conf_threshold=0.5, max_age=15)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, tracks = tracker.update(frame)
        cv2.imshow("Player Tracker", annotated_frame)
        print("Tracks:", tracks)
        if cv2.waitKey(1) == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()
