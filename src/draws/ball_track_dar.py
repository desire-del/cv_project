from src.draws.utils import draw_traingle
import cv2
import numpy as np
from collections import defaultdict

class BallTracksDrawer:
    """
    A class that draws the ball's trajectory and direction on video frames.
    """

    def __init__(self, trail_length=15, ball_color=(0, 255, 0)):
        """
        Args:
            trail_length (int): Number of past positions to show as trail.
            ball_color (tuple): BGR color for the ball.
        """
        self.trail_length = trail_length
        self.ball_color = ball_color
        self.trail_history = defaultdict(list)

    def draw_trail(self, frame, trail, color):
        """
        Draws a trail showing past ball positions.

        Args:
            frame (np.ndarray): The frame to draw on.
            trail (list): List of (x, y) positions.
            color (tuple): BGR color for the trail.
        """
        for i, point in enumerate(reversed(trail[-self.trail_length:])):
            alpha = (i + 1) / self.trail_length
            cv2.circle(frame, point, 3, [int(c * alpha) for c in color], -1)

    def draw(self, video_frames, tracks):
        """
        Draws ball markers and trail on each frame.

        Args:
            video_frames (list): List of frames (np.ndarray).
            tracks (list): List of dicts with ball detections per frame.

        Returns:
            list: List of annotated frames.
        """
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            ball_dict = tracks[frame_num]

            for track_id, ball in ball_dict.items():
                bbox = ball.get("bbox")
                if not bbox:
                    continue

                x, y, w, h = bbox
                center = (int(x + w / 2), int(y + h / 2))

                # Trail update
                self.trail_history[track_id].append(center)

                # Draw trail
                self.draw_trail(frame, self.trail_history[track_id], self.ball_color)

                # Draw triangle marker on ball
                frame = draw_traingle(frame, bbox, self.ball_color)

                # Optional: direction arrow
                if len(self.trail_history[track_id]) > 1:
                    prev = self.trail_history[track_id][-2]
                    cv2.arrowedLine(frame, prev, center, self.ball_color, 2, tipLength=0.4)

            output_video_frames.append(frame)

        return output_video_frames
