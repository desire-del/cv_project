from src.tracks.player_tracker import PlayerTracker
from src.draws.draw_player import PlayerTracksDrawer
from src.draws.ball_track_dar import BallTracksDrawer
from src.utils import read_video, save_video
from src.teams.teams_assigner import TeamAssigner
from src.tracks.ball_tracker import BallTracker
from src.ball_aquisition.ball_aquisition_detector import BallAquisitionDetector
from src.draws.teams_ball_pos_draw import TeamBallControlDrawer
from src.draws.passes_interceptions_draw import PassInterceptionDrawer
from src.passes.passes_interceptions import PassAndInterceptionDetector
from src.court_keypoint_detector.court_keypoint_detector import CourtKeypointDetector
from src.draws.court_key_points_drawer import CourtKeypointDrawer
from src.tactic_view.tactic_view_converter import TacticalViewConverter
from src.draws.tactic_viewer_drawer import TacticalViewDrawer
from src.speed_and_distance_calculator import SpeedAndDistanceCalculator
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


ball_acquisition_detector = BallAquisitionDetector()
ball_acquisition = ball_acquisition_detector.detect_ball_possession(player_tracks=player_tracks,ball_tracks=ball_tracks_result)

passes_interception_detector = PassAndInterceptionDetector()
passes = passes_interception_detector.detect_passes(ball_acquisition=ball_acquisition,
                                                     player_assignment=player_teams)

interceptions = passes_interception_detector.detect_interceptions(ball_acquisition=ball_acquisition,
                                                              player_assignment=player_teams)

court_keypoint_detector = CourtKeypointDetector()
court_keypoints = court_keypoint_detector.detect_keypoints(frames=frames, read_from_stub=True, stub_path="cache/court_keypoints.pkl")

tactical_view_converter = TacticalViewConverter(court_image_path="data/basketball_court.png")

court__keypoints_per_frame = tactical_view_converter.validate_keypoints(court_keypoints)
tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court__keypoints_per_frame, player_tracks)

player_drawer = PlayerTracksDrawer()
ball_drawer = BallTracksDrawer()
ball_possession_drawer = TeamBallControlDrawer(team_colors={1: [255, 245, 238], 2: [128, 0, 0]})
pass_interception_drawer = PassInterceptionDrawer(team_colors={1: [255, 245, 238], 2: [128, 0, 0]})
court_keypoint_drawer = CourtKeypointDrawer()
tactical_view_drawer = TacticalViewDrawer(team_1_color=[255, 245, 238], team_2_color=[128, 0, 0])


output_frame = player_drawer.draw(video_frames=frames,
                   tracks=player_tracks,
                    player_assignment=player_teams,
                   ball_acquisition=ball_acquisition)

output_frame = ball_drawer.draw(video_frames=output_frame,
                                 tracks=ball_tracks_result)

output_frame = tactical_view_drawer.draw(output_frame,
                                        tactical_view_converter.court_image_path,
                                        tactical_view_converter.width,
                                        tactical_view_converter.height,
                                        tactical_view_converter.key_points,
                                        tactical_player_positions,
                                        player_teams,
                                        ball_acquisition
                                        )

output_frame = ball_possession_drawer.draw(video_frames=output_frame,
                                            player_assignment=player_teams,
                                            ball_acquisition=ball_acquisition)

output_frame = pass_interception_drawer.draw(video_frames=output_frame,
                                               passes=passes,
                                               interceptions=interceptions)

output_frame = court_keypoint_drawer.draw(frames=output_frame,
                                            court_keypoints=court_keypoints)

save_video(frames=output_frame,
           path="data/videos/video_1_output.mp4",
           fps=fps)