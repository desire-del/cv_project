# src/keypoints/court_keypoint_detector.py

from ultralytics import YOLO
import numpy as np
import sys
sys.path.append('../../') 
from src.utils import read_stub, save_stub


class CourtKeypointDetector:
    """
    Detecte les keypoints du terrain (ex : lignes de terrain de basket) à partir d’un modèle YOLOv8 keypoints.
    """

    def __init__(self, model_path="models/court_keypoints.pt"):
        self.model = YOLO(model_path)

    def detect_keypoints(self, frames, read_from_stub=False, stub_path=None):
        """
        Détecte les keypoints sur chaque frame avec gestion de cache.

        Args:
            frames (List[np.ndarray]): Images de la vidéo.
            read_from_stub (bool): Si True, lire depuis le cache si disponible.
            stub_path (str): Chemin vers le fichier de cache (pickle).

        Returns:
            List[List[np.ndarray]]: Liste des keypoints par frame.
        """
        # Lecture du cache
        cached = read_stub(read_from_stub, stub_path)
        if cached is not None and len(cached) == len(frames):
            return cached

        keypoints_per_frame = []

        print("📍 Détection des keypoints YOLO...")

        batch_size = 16
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            results = self.model.predict(batch, conf=0.4, verbose=False)

            for r in results:
                if r.keypoints is not None:
                    kpts = r.keypoints.xy.cpu().numpy()  # [N, K, 2]
                    keypoints_per_frame.append(kpts.tolist())
                else:
                    keypoints_per_frame.append([])

        save_stub(stub_path, keypoints_per_frame)
        return keypoints_per_frame
