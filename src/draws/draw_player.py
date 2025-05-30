from src.draws.utils import draw_ellipse, draw_triangle
import cv2
from collections import defaultdict

class PlayerTracksDrawer:
    """
    A class responsible for drawing player tracks and ball possession indicators on video frames.

    Attributes:
        default_player_team_id (int): Default team ID used when a player's team is not specified.
        team_1_color (list): RGB color used to represent Team 1 players.
        team_2_color (list): RGB color used to represent Team 2 players.
        trail_length (int): Number of past positions to keep for each player to draw trails.
    """
    def __init__(self, team_1_color=[255, 245, 238], team_2_color=[128, 0, 0], trail_length=10):
        self.default_player_team_id = 1
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color
        self.trail_length = trail_length
        self.trail_history = defaultdict(list)

    def draw_trail(self, frame, trail, color):
        """
        Draws a fading trail behind a player using a list of past positions.
        """
        for i, point in enumerate(reversed(trail[-self.trail_length:])):
            alpha = (i + 1) / self.trail_length
            faded_color = [int(c * alpha) for c in color]
            cv2.circle(frame, point, 3, faded_color, -1)

    def draw(self, video_frames, tracks, player_assignment, ball_acquisition):
        """
        Draw player tracks, trails and ball possession indicators on a list of video frames.

        Args:
            video_frames (list): List of frames (np.array) on which to draw.
            tracks (list): List of dicts with player tracking info per frame.
            player_assignment (list): List of dicts indicating team for each player per frame.
            ball_aquisition (list): List indicating which player has the ball per frame.

        Returns:
            list: Frames with drawings applied.
        """

        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks[frame_num]
            player_assignment_for_frame = player_assignment[frame_num]
            player_id_has_ball = ball_acquisition[frame_num]

            for track_id, player in player_dict.items():
                # Récupérer le team_id (1 par défaut)
                
                team_id = player_assignment_for_frame.get(track_id, self.default_player_team_id)

                # Couleur selon l’équipe
                color = self.team_1_color if team_id == 1 else self.team_2_color

                # Calcul du centre du bbox pour la trace
                bbox = player["bbox"]
                x1, y1, x2, y2 = bbox
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                # Ajouter le centre au trail_history (trajectoire)
                self.trail_history[track_id].append(center)

                # Dessiner la trace (trail)
                self.draw_trail(frame, self.trail_history[track_id], color)

                # Dessiner le joueur (ellipse + id)
                frame = draw_ellipse(frame, bbox, color, int(track_id))

                # Dessiner un triangle rouge si le joueur a le ballon
                if track_id == player_id_has_ball:
                    frame = draw_triangle(frame, bbox, (0, 0, 255))

            output_video_frames.append(frame)

        return output_video_frames
