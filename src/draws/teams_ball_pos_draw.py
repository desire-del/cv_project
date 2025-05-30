import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

class TeamBallControlDrawer:
    """
    Classe pour calculer et afficher les statistiques de contrôle de balle par équipe sur des frames vidéo.
    """

    def __init__(self,
                 overlay_alpha: float = 0.6,
                 overlay_color: tuple = (0, 0, 0),  # fond noir semi-transparent
                 font_scale: float = 0.8,
                 font_thickness: int = 2,
                 text_color: tuple = (255, 255, 255),
                 team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None):
        self.overlay_alpha = overlay_alpha
        self.overlay_color = overlay_color
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.text_color = text_color

        # Couleurs par équipe : dict {team_id: (B,G,R)}
        if team_colors is None:
            # Couleurs par défaut : équipe 1 rouge, équipe 2 bleu
            self.team_colors = {
                1: (0, 0, 255),
                2: (255, 0, 0)
            }
        else:
            self.team_colors = team_colors

    def get_team_ball_control(
        self,
        player_assignment: List[Dict[int, int]],
        ball_acquisition: List[int]
    ) -> np.ndarray:
        """
        Calcule l'équipe en contrôle de balle pour chaque frame.

        Args:
            player_assignment: Liste où chaque élément est un dict {player_id: team_id} par frame.
            ball_acquisition: Liste des player_id qui contrôlent la balle à chaque frame (-1 si aucun).

        Returns:
            np.ndarray d'entiers indiquant le contrôle de balle par frame:
                1 pour équipe 1,
                2 pour équipe 2,
                -1 pour aucun contrôle.
        """
        team_ball_control = []
        for pa_frame, ba_frame in zip(player_assignment, ball_acquisition):
            if ba_frame == -1 or ba_frame not in pa_frame:
                team_ball_control.append(-1)
            else:
                team = pa_frame[ba_frame]
                if team in (1, 2):
                    team_ball_control.append(team)
                else:
                    team_ball_control.append(-1)
        return np.array(team_ball_control, dtype=np.int8)

    def draw(
        self,
        video_frames: List[np.ndarray],
        player_assignment: List[Dict[int, int]],
        ball_acquisition: List[int]
    ) -> List[np.ndarray]:
        """
        Dessine les statistiques de contrôle de balle sur chaque frame.

        Args:
            video_frames: Liste de frames (ndarray) sur lesquelles dessiner.
            player_assignment: Liste de dicts {player_id: team_id} par frame.
            ball_acquisition: Liste des player_id en possession par frame.

        Returns:
            Liste de frames avec dessin superposé.
        """
        team_ball_control = self.get_team_ball_control(player_assignment, ball_acquisition)

        output_frames = []
        total_frames = len(team_ball_control)
        for i, frame in enumerate(video_frames):
            # Protection si moins de frames que prévu
            if i >= total_frames:
                output_frames.append(frame)
                continue
            frame_drawn = self.draw_frame(frame.copy(), i, team_ball_control)
            output_frames.append(frame_drawn)
        return output_frames

    def draw_frame(
        self,
        frame: np.ndarray,
        frame_num: int,
        team_ball_control: np.ndarray
    ) -> np.ndarray:
        """
        Dessine un overlay avec un fond sombre semi-transparent, texte et barres colorées de contrôle de balle.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Rectangle semi-transparent foncé
        rect_x1 = int(w * 0.50)
        rect_y1 = int(h * 0.65)
        rect_x2 = int(w * 0.99)
        rect_y2 = int(h * 0.90)
        rect_width = rect_x2 - rect_x1
        rect_height = rect_y2 - rect_y1

        # Fond noir transparent
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), self.overlay_color, -1)
        cv2.addWeighted(overlay, self.overlay_alpha, frame, 1 - self.overlay_alpha, 0, frame)

        # Calcul des stats
        ball_control_slice = team_ball_control[:frame_num + 1]
        total = ball_control_slice.shape[0]
        if total == 0:
            team1_pct = 0.0
            team2_pct = 0.0
        else:
            team1_pct = np.sum(ball_control_slice == 1) / total * 100
            team2_pct = np.sum(ball_control_slice == 2) / total * 100

        # Positions pour les barres et textes
        bar_x = rect_x1 + int(rect_width * 0.25)
        bar_y_team1 = rect_y1 + int(rect_height * 0.3)
        bar_y_team2 = rect_y1 + int(rect_height * 0.65)
        bar_height = 20
        max_bar_width = int(rect_width * 0.7)

        # Couleurs équipe personnalisées
        color_team1 = self.team_colors.get(1, (0, 0, 255))  # Rouge par défaut
        color_team2 = self.team_colors.get(2, (255, 0, 0))  # Bleu par défaut
        text_color = self.text_color

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.font_scale
        thickness = self.font_thickness

        # Texte avec légende colorée (cercle)
        circle_radius = 10
        circle_x = rect_x1 + int(rect_width * 0.1)
        # Team 1
        cv2.circle(frame, (circle_x, bar_y_team1), circle_radius, color_team1, -1)
        cv2.putText(frame, f"Team 1 Ball Control: {team1_pct:.1f}%", 
                    (bar_x, bar_y_team1 + 10), font, font_scale, text_color, thickness, cv2.LINE_AA)
        # Team 2
        cv2.circle(frame, (circle_x, bar_y_team2), circle_radius, color_team2, -1)
        cv2.putText(frame, f"Team 2 Ball Control: {team2_pct:.1f}%", 
                    (bar_x, bar_y_team2 + 10), font, font_scale, text_color, thickness, cv2.LINE_AA)

        # Barres horizontales proportionnelles
        team1_bar_width = int(max_bar_width * (team1_pct / 100))
        team2_bar_width = int(max_bar_width * (team2_pct / 100))

        # Dessin des barres (avec coins arrondis simplifiés)
        def draw_rounded_bar(img, start_x, start_y, width, height, color):
            cv2.rectangle(img, (start_x, start_y), (start_x + width, start_y + height), color, -1)
            radius = height // 2
            if width > 0:
                cv2.circle(img, (start_x + radius, start_y + radius), radius, color, -1)
                cv2.circle(img, (start_x + width - radius, start_y + radius), radius, color, -1)

        draw_rounded_bar(frame, bar_x, bar_y_team1 + 25, team1_bar_width, bar_height, color_team1)
        draw_rounded_bar(frame, bar_x, bar_y_team2 + 25, team2_bar_width, bar_height, color_team2)

        return frame
