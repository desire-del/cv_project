from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel
import torch
import sys
import logging
from typing import Tuple, List, Dict, Optional

sys.path.append('../../')
from utils import read_stub, save_stub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeamAssigner:
    def __init__(self,
                 team_1_class_name: str = "white shirt",
                 team_2_class_name: str = "dark blue shirt"):
        self.team_colors: Dict[int, str] = {}
        self.player_team_dict: Dict[int, int] = {}        
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name
        self.model = None
        self.processor = None

    def load_model(self):
        try:
            self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
            self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
            logger.info("Fashion-CLIP model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_player_color(self,frame,bbox):
        """
        Analyzes the jersey color of a player within the given bounding box.

        Args:
            frame (numpy.ndarray): The video frame containing the player.
            bbox (tuple): Bounding box coordinates of the player.

        Returns:
            str: The classified jersey color/description.
        """
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        print(f"Processing bbox: {bbox}, image shape: {image.shape}")

        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        image = pil_image

        classes = [self.team_1_class_name, self.team_2_class_name]

        inputs = self.processor(text=classes, images=image, return_tensors="pt", padding=True)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1) 


        class_name=  classes[probs.argmax(dim=1)[0]]

        return class_name


    def get_player_team(self, frame, player_bbox: Tuple[int, int, int, int], player_id: int) -> int:
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        color = self.get_player_color(frame, player_bbox)
        if color == self.team_1_class_name:
            team_id = 1
        else:
            team_id = 2

        self.player_team_dict[player_id] = team_id
        return team_id

    def get_player_teams_across_frames(self,
                                       video_frames: List,
                                       player_tracks: List[Dict[int, dict]],
                                       read_from_stub: bool = False,
                                       stub_path: Optional[str] = None) -> List[Dict[int, int]]:

        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None and len(player_assignment) == len(video_frames):
            logger.info("Using cached team assignments.")
            return player_assignment

        self.load_model()

        player_assignment = []
        for frame_num, player_track in enumerate(player_tracks):
            player_assignment.append({})
            if frame_num % 50 == 0:
                self.player_team_dict.clear()

            for player_id, track in player_track.items():
                team = self.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                player_assignment[frame_num][player_id] = team

        save_stub(stub_path, player_assignment)
        return player_assignment
