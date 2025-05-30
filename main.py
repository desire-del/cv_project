from src.tracks.player_tracker import PlayerTracker
from src.draws.draw_player import PlayerTracksDrawer
from src.utils import read_video, save_video
from src.teams.teams_assigner import TeamAssigner
import cv2

model_path = "models/players_detection_model.pt"
video_path = "data/videos/video_1.mp4"

frames, fps = read_video(video_path)

tracker = PlayerTracker(model_path=model_path,
                        max_age=15,
                        conf_threshold=0.5)

player_tracks = tracker.track_players(frames=frames, cache_path="cache/stub.pkl", use_cache=True)

team_assigner = TeamAssigner()
player_teams = team_assigner.get_player_teams_across_frames(video_frames=frames,
                                                              player_tracks=player_tracks,
                                                              read_from_stub=True,
                                                              stub_path="cache/team_assignments.pkl")
print(player_teams)

#drawer = PlayerTracksDrawer()

#drawer.draw(frames, player_tracks)