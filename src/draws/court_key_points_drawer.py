import supervision as sv
import numpy as np

class CourtKeypointDrawer:
    """
    Classe de dessin des keypoints du terrain sur une séquence de frames.
    """

    def __init__(self):
        self.keypoint_color = '#ff2c2c'
        self.vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            radius=8
        )
        self.vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1
        )

    def draw(self, frames, court_keypoints):
        """
        Dessine les keypoints du terrain sur les frames.

        Args:
            frames (list): Liste de frames vidéo (np.ndarray).
            court_keypoints (list): Liste de keypoints par frame, chacun étant une liste de (x, y).

        Returns:
            list: Frames annotées.
        """
        output_frames = []

        for index, frame in enumerate(frames):
            annotated_frame = frame.copy()
            keypoints = court_keypoints[index]

            if not keypoints:
                output_frames.append(annotated_frame)
                continue

            # Si keypoints est un Tensor, on le convertit
            if hasattr(keypoints, "cpu"):
                keypoints = keypoints.cpu().numpy()
            else:
                keypoints = np.array(keypoints)

            keypoints_obj = sv.KeyPoints(keypoints)

            annotated_frame = self.vertex_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints_obj
            )

            annotated_frame = self.vertex_label_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints_obj
            )

            output_frames.append(annotated_frame)

        return output_frames
