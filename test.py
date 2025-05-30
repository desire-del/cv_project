from ultralytics import YOLO

# Charger le modèle YOLO (détecteur de balle)
model_path = "models/players_detection_model.pt"
model = YOLO(model_path)

# Appliquer la détection sur la vidéo (streaming image par image)
results = model.predict(
    source="data/videos/video_1.mp4",  # Chemin vers ta vidéo
    conf=0.5,                          # Seuil de confiance                       # Pour traiter frame par frame
    show=True,                         # Pour afficher pendant l'exécution
    save=True                          # Pour enregistrer la vidéo
)
for result in results:
    # Afficher les résultats de détection pour chaque frame
    print(result.boxes)  # Affiche les boîtes de détection
    print(result.masks)  # Affiche les masques de détection (si disponibles)
    print(result.names)  # Affiche les noms des classes détectées