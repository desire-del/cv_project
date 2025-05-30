from src.draws.utils import draw_triangle
import cv2
import numpy as np
from collections import defaultdict

class BallTracksDrawer:
    """
    A class that draws the ball's trajectory and direction on video frames.
    """

    def __init__(self, trail_length=15, ball_color=(0, 255, 0)):
        self.trail_length = trail_length
        self.ball_color = ball_color
        self.trail_history = defaultdict(list)

    def draw_trail(self, frame, trail, color):
        for i, point in enumerate(reversed(trail[-self.trail_length:])):
            alpha = (i + 1) / self.trail_length
            cv2.circle(frame, point, 3, [int(c * alpha) for c in color], -1)

    def draw(self, video_frames, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            ball_dict = tracks[frame_num]

            for track_id, ball in ball_dict.items():
                bbox = ball.get("bbox")  # Expected format: [x1, y1, x2, y2]
                if not bbox:
                    continue

                x1, y1, x2, y2 = bbox
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                self.trail_history[track_id].append(center)
                self.draw_trail(frame, self.trail_history[track_id], self.ball_color)
                frame = draw_triangle(frame, bbox, self.ball_color)

                # Draw triangle marker on the ball
                frame = draw_triangle(frame, bbox, self.ball_color)

                # Optional: direction arrow
                if len(self.trail_history[track_id]) > 1:
                    prev = self.trail_history[track_id][-2]
                    cv2.arrowedLine(frame, prev, center, self.ball_color, 2, tipLength=0.4)

            output_video_frames.append(frame)

        return output_video_frames
