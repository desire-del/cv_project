import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

class PassInterceptionDrawer:
    """
    Classe pour calculer et dessiner les statistiques cumulées
    de passes et interceptions par équipe sur des frames vidéo.
    """

    def __init__(
        self,
        overlay_alpha: float = 0.6,
        overlay_color: Tuple[int, int, int] = (0, 0, 0),  # fond noir semi-transparent
        font_scale: float = 0.7,
        font_thickness: int = 2,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        max_passes_interceptions: Optional[int] = None  # max pour barres proportionnelles
    ):
        self.overlay_alpha = overlay_alpha
        self.overlay_color = overlay_color
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.text_color = text_color

        # Couleurs par équipe (BGR)
        self.team_colors = team_colors or {
            1: (0, 0, 255),   # rouge
            2: (255, 0, 0)    # bleu
        }

        # Pour normaliser la taille des barres (max passes + interceptions)
        self.max_val = max_passes_interceptions
        self.cumulative_stats = []

    def prepare_stats(self, passes: List[int], interceptions: List[int]) -> None:
        """
        Calcule les statistiques cumulées pour chaque frame.

        Args:
            passes: Liste d'événements passes par frame (0: aucune, 1: équipe 1, 2: équipe 2).
            interceptions: Liste d'événements interceptions par frame (0: aucune, 1: équipe 1, 2: équipe 2).
        """
        team1_passes = 0
        team2_passes = 0
        team1_interceptions = 0
        team2_interceptions = 0
        self.cumulative_stats = []

        for p, i in zip(passes, interceptions):
            if p == 1:
                team1_passes += 1
            elif p == 2:
                team2_passes += 1

            if i == 1:
                team1_interceptions += 1
            elif i == 2:
                team2_interceptions += 1

            self.cumulative_stats.append((team1_passes, team2_passes, team1_interceptions, team2_interceptions))

        if self.max_val is None:
            # Définir max_val pour la mise à l'échelle des barres
            max_passes = max(team1_passes, team2_passes)
            max_interceptions = max(team1_interceptions, team2_interceptions)
            self.max_val = max_passes + max_interceptions
            if self.max_val == 0:
                self.max_val = 1  # éviter division par zéro

    def draw(
        self,
        video_frames: List[np.ndarray],
        passes: List[int],
        interceptions: List[int]
    ) -> List[np.ndarray]:
        """
        Dessine les statistiques cumulées sur chaque frame.

        Args:
            video_frames: Liste de frames (np.ndarray).
            passes: Liste des passes par frame.
            interceptions: Liste des interceptions par frame.

        Returns:
            Liste des frames avec overlay statistique.
        """
        self.prepare_stats(passes, interceptions)

        output_frames = []
        for idx, frame in enumerate(video_frames):
            frame_copy = frame.copy()
            if idx >= len(self.cumulative_stats):
                output_frames.append(frame_copy)
                continue

            stats = self.cumulative_stats[idx]
            frame_with_stats = self.draw_frame(frame_copy, idx, stats)
            output_frames.append(frame_with_stats)

        return output_frames

    def draw_frame(
        self,
        frame: np.ndarray,
        frame_num: int,
        stats: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Dessine l'overlay avec stats cumulées (passes et interceptions).

        Args:
            frame: Frame image.
            frame_num: Index de la frame (non utilisé ici, mais pour extension possible).
            stats: Tuple (team1_passes, team2_passes, team1_interceptions, team2_interceptions).

        Returns:
            Frame modifiée.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Rectangle semi-transparent fond sombre
        rect_x1 = int(w * 0.05)
        rect_y1 = int(h * 0.65)
        rect_x2 = int(w * 0.45)
        rect_y2 = int(h * 0.90)

        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), self.overlay_color, thickness=-1)
        cv2.addWeighted(overlay, self.overlay_alpha, frame, 1 - self.overlay_alpha, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        line_spacing = int(h * 0.05)
        margin_x = rect_x1 + 10
        base_y = rect_y1 + 25

        team1_passes, team2_passes, team1_interceptions, team2_interceptions = stats

        # Texte + barres pour chaque équipe et type d'event
        def draw_stat_line(img, x, y, label, count, color):
            text = f"{label}: {count}"
            # Ombre texte
            cv2.putText(img, text, (x + 1, y + 1), font, self.font_scale, (0, 0, 0), self.font_thickness + 1, cv2.LINE_AA)
            cv2.putText(img, text, (x, y), font, self.font_scale, color, self.font_thickness, cv2.LINE_AA)

            

        # Draw team 1 stats (rouge)
        color_team1 = self.team_colors.get(1, (0, 0, 255))
        draw_stat_line(frame, margin_x, base_y, "Team 1 Passes", team1_passes, color_team1)
        draw_stat_line(frame, margin_x, base_y + line_spacing, "Team 1 Interceptions", team1_interceptions, color_team1)

        # Draw team 2 stats (bleu)
        color_team2 = self.team_colors.get(2, (255, 0, 0))
        draw_stat_line(frame, margin_x, base_y + 2 * line_spacing, "Team 2 Passes", team2_passes, color_team2)
        draw_stat_line(frame, margin_x, base_y + 3 * line_spacing, "Team 2 Interceptions", team2_interceptions, color_team2)

        return frame
