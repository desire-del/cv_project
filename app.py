import streamlit as st
import tempfile
import os
import cv2
import sys
#sys.path.append('../')
from src.utils import read_video, save_video
from src.tracks.player_tracker import PlayerTracker
from src.tracks.ball_tracker import BallTracker
from src.teams.teams_assigner import TeamAssigner
from src.ball_aquisition.ball_aquisition_detector import BallAquisitionDetector
from src.passes.passes_interceptions import PassAndInterceptionDetector
from src.court_keypoint_detector.court_keypoint_detector import CourtKeypointDetector
from src.tactic_view.tactic_view_converter import TacticalViewConverter
from src.draws.draw_player import PlayerTracksDrawer
from src.draws.ball_track_dar import BallTracksDrawer
from src.draws.teams_ball_pos_draw import TeamBallControlDrawer
from src.draws.passes_interceptions_draw import PassInterceptionDrawer
from src.draws.court_key_points_drawer import CourtKeypointDrawer
from src.draws.tactic_viewer_drawer import TacticalViewDrawer

def main():
    st.set_page_config(layout="wide", page_title="Analyse Tactique Vidéo", page_icon="🏀")
    with open("frontend/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Header with logo
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("data/logo.png", width=120)
    with col2:
        st.title("Analyse Vidéo de Matchs de Basketball avec Détection Automatique")
        st.markdown("**Projet Universitaire** - Analyse tactique assistée par l'IA")

    st.markdown("---")

    uploaded_video = st.file_uploader("Téléverser une vidéo", type=["mp4", "mov", "avi"])
    default_video_path = "data/videos/video_1.mp4"  # 🟡 Remplace avec ton vrai chemin

    st.markdown("### Choisissez les couleurs des équipes")

    team_color_cols = st.columns(2)
    with team_color_cols[0]:
        team1_color = st.color_picker("Couleur Équipe 1", "#FFF5EE", key="team1")
        st.markdown(f"<div style='height:20px;width:50px;background-color:{team1_color};border:1px solid #000'></div>", unsafe_allow_html=True)

    with team_color_cols[1]:
        team2_color = st.color_picker("Couleur Équipe 2", "#800000", key="team2")
        st.markdown(f"<div style='height:20px;width:50px;background-color:{team2_color};border:1px solid #000'></div>", unsafe_allow_html=True)

    # 🔁 Si aucune vidéo n'est uploadée, utiliser la vidéo par défaut
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name
    else:
        st.warning("Aucune vidéo uploadée. La vidéo par défaut sera utilisée.")
        video_path = default_video_path

    st.info("Traitement en cours...")

    output_video_path = process_pipeline(
        video_path=video_path,
        team_colors={1: hex_to_bgr(team1_color), 2: hex_to_bgr(team2_color)}
    )

    st.success("Traitement terminé avec succès")

    st.markdown("### Résultats Vidéo")

    input_col, output_col = st.columns(2)

    with input_col:
        st.subheader("Vidéo Originale")
        st.video(video_path)

    with output_col:
        st.subheader("Vidéo Annotée")
        st.video(output_video_path)

        with open(output_video_path, 'rb') as f:
            st.download_button("📥 Télécharger la vidéo annotée", f, file_name="output.mp4")

    


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) for i in (4, 2, 0)]  # R, G, B -> BGR


def process_pipeline(video_path, team_colors):
    model_path = "models/players_detection_model.pt"
    

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
    output_path = "data/videos/video_1_output.mp4"
    save_video(frames=output_frame,
            path=output_path,
            fps=fps)
    
    return output_path


if __name__ == "__main__":
    main()
