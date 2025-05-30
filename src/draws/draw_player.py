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
from src.draws.utils import draw_ellipse, draw_triangle
import cv2
import numpy as np
from collections import defaultdict

class PlayerTracksDrawer:
    """
    A class that draws visually rich player tracks, team info, and ball possession indicators on frames.
    """

    def __init__(self, team_1_color=[0, 255, 255], team_2_color=[255, 0, 255], trail_length=10):
        self.default_player_team_id = 1
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color
        self.trail_length = trail_length
        self.trail_history = defaultdict(list)

    def draw_player(self, frame, bbox, color, player_id, highlight=False):
        """
        Draws a player with an ID and optionally a glowing halo if in ball possession.
        bbox: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        # Glow if highlight
        if highlight:
            cv2.circle(frame, center, int(w / 1.5), (0, 0, 255), 10, cv2.LINE_AA)

        # Ellipse around player
        cv2.ellipse(frame, center, (int(w / 2), int(h / 2)), 0, 0, 360, color, 2)

        # ID label with translucent background
        overlay = frame.copy()
        label = f"#{player_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        text_x, text_y = x1, y1 - 10
        cv2.rectangle(overlay, (text_x, text_y - th - 4), (text_x + tw + 6, text_y + 4), color, -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, label, (text_x + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return frame

    def draw_trail(self, frame, trail, color):
        for i, point in enumerate(reversed(trail[-self.trail_length:])):
            alpha = (i + 1) / self.trail_length
            cv2.circle(frame, point, 3, [int(c * alpha) for c in color], -1)

    def draw(self, video_frames, tracks, player_assignment, ball_acquisition):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks[frame_num]
            team_dict = player_assignment[frame_num]
            ball_holder_id = ball_acquisition[frame_num]

            for track_id, player in player_dict.items():
                team_id = team_dict.get(track_id, self.default_player_team_id)
                color = self.team_1_color if team_id == 1 else self.team_2_color
                bbox = player["bbox"]  # expected as [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                self.trail_history[track_id].append(center)
                self.draw_trail(frame, self.trail_history[track_id], color)

                highlight = (track_id == ball_holder_id)
                frame = self.draw_player(frame, bbox, color, track_id, highlight=highlight)

                if len(self.trail_history[track_id]) > 1:
                    prev = self.trail_history[track_id][-2]
                    cv2.arrowedLine(frame, prev, center, color, 1, tipLength=0.3)

            output_video_frames.append(frame)

        return output_video_frames
