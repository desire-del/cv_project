import cv2
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.utils import read_video, save_video
from src.speed_and_distance_calculator import SpeedAndDistanceCalculator
from src.draws.speed_and_distance_drawer import SpeedAndDistanceDrawer

def main():
    # === Paramètres
    video_path = "data/videos/video_1.mp4"
    output_path = "output/match_annotated.mp4"
    model_path = "models/players_detection_model.pt"

    # === Paramètres terrain
    width_meters = 28.0  # terrain de basket officiel
    height_meters = 15.0

    # === Lecture de la vidéo
    frames, fps = read_video(video_path)
    height, width = frames[0].shape[:2]

    # === Initialiser le tracker
    tracker = DeepSort(max_age=30)

    all_tracks = []

    for frame in frames:
        # ⚠️ Adapter ici si tu utilises YOLO pour obtenir les détections
        detections = []  # Format: [(x1, y1, x2, y2, conf, class_id), ...]
        tracks = tracker.update_tracks(detections, frame=frame)
        all_tracks.append(tracks)

    # === Calcul distance & vitesse
    calculator = SpeedAndDistanceCalculator(
        width_in_pixels=width,
        height_in_pixels=height,
        width_in_meters=width_meters,
        height_in_meters=height_meters
    )

    distances = calculator.calculate_distance(all_tracks)
    speeds = calculator.calculate_speed(distances, fps=fps)

    # === Annotation
    drawer = SpeedAndDistanceDrawer()
    annotated_frames = drawer.draw(frames, all_tracks, distances, speeds)

    os.makedirs("output", exist_ok=True)
    save_video(annotated_frames, output_path, fps)
    print(f"✅ Vidéo annotée sauvegardée dans : {output_path}")

if __name__ == "__main__":
    main()
