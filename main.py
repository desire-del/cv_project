from src.tracks.player_tracker import PlayerTracker
from src.draws.draw_player import PlayerTracksDrawer
from src.draws.ball_track_dar import BallTracksDrawer
from src.utils import read_video, save_video
from src.teams.teams_assigner import TeamAssigner
from src.tracks.ball_tracker import BallTracker
from src.ball_aquisition.ball_aquisition_detector import BallAquisitionDetector
import cv2

model_path = "models/players_detection_model.pt"
video_path = "data/videos/video_1.mp4"

frames, fps = read_video(video_path)

tracker = PlayerTracker(model_path=model_path,
                        max_age=15,
                        conf_threshold=0.5)

ball_tracks = BallTracker(model_path=model_path, max_age=20)
ball_tracks_result = ball_tracks.get_object_tracks(frames=frames, read_from_stub=True, stub_path="cache/ball_tracks.pkl")
ball_tracks_result = ball_tracks.remove_wrong_detections(ball_tracks_result, max_distance=25)
ball_tracks_result = ball_tracks.interpolate_ball_positions(ball_tracks_result)


player_tracks = tracker.track_players(frames=frames, cache_path="cache/stub.pkl", use_cache=True)

team_assigner = TeamAssigner()
player_teams = team_assigner.get_player_teams_across_frames(video_frames=frames,
                                                              player_tracks=player_tracks,
                                                              read_from_stub=True,
                                                              stub_path="cache/team_assignments.pkl")
print(ball_tracks_result)
print(player_teams)

ball_acquisition_detector = BallAquisitionDetector()
ball_acquisition = ball_acquisition_detector.detect_ball_possession(player_tracks=player_tracks,ball_tracks=ball_tracks_result)


player_drawer = PlayerTracksDrawer()
ball_drawer = BallTracksDrawer()

output_frame = player_drawer.draw(video_frames=frames,
                   tracks=player_tracks,
                    player_assignment=player_teams,
                   ball_acquisition=ball_acquisition)

output_frame = ball_drawer.draw(video_frames=output_frame,
                                 tracks=ball_tracks_result)
save_video(frames=output_frame,
           path="data/videos/video_1_output.mp4",
           fps=fps)