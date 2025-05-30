from src.tracks.player_tracker import PlayerTracker
from src.draws import draw_player
from src.tracks.ball_tracker1 import BallTracker
from src.draws.ball_track_dar import BallTracksDrawer
from src.utils import read_video, save_video
import cv2

model_path = "models/players_detection_model.pt"
video_path = "data/videos/video_1.mp4"

frames, fps = read_video(video_path)

tracker = PlayerTracker(model_path=model_path,
                        max_age=15,
                        conf_threshold=0.5)

player_tracks = tracker.track_players(frames=frames, cache_path="cache/stub.pkl", use_cache=True)

#drawer = PlayerTracksDrawer()

#drawer.draw(frames, player_tracks)