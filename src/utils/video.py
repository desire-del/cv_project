import cv2

def read_video(path):
    """
    Read a video from the given path and return a list of frames and FPS.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps


def save_video(frames, path, fps=30):
    """
    Save a list of frames to a video file.
    """
    if not frames:
        raise ValueError("No frames to write.")

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use 'XVID' or 'avc1' if needed
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)
    
    out.release()