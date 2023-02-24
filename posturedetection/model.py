import mediapipe as mp
import numpy as np
from .utils import get_angle_from_vertical, get_angle_at_joint, angle_in_range

holistic = mp.solutions.holistic
triplets = {
            'left_shoulder': ('left_shoulder', 'left_elbow', 'left_hip'),
            'right_shoulder': ('right_shoulder', 'right_elbow', 'right_hip'),
            'left_elbow': ('left_elbow', 'left_wrist', 'left_shoulder'),
            'right_elbow': ('right_elbow', 'right_wrist', 'right_shoulder')
        }

class PoseDetector:
    def __init__(self, dim:tuple[int], mirror :bool=False, angle_margin :float=10):
        self.dim = dim
        self.mirror = mirror
        self.pose_names = holistic.PoseLandmark._member_names_
        self.holistic = holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.drawing = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles
        self.landmarks = None
        self.angle_margin = angle_margin
        self.pose_requirements = []
        self.mirror_requirements = []
        self.angles = {}

    def detect(self, frame: np.ndarray) -> list:
        landmarks = self.holistic.process(frame).pose_landmarks
        self.landmarks = landmarks
        return landmarks
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        self.drawing.draw_landmarks(
            frame, 
            self.landmarks, 
            holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.styles.get_default_pose_landmarks_style()
        )

    def get_landmark_id(self, name: str) -> int:
        return [i for i, j in enumerate(self.pose_names) if j == name][0]
    
    def get_landmark(self, name:str):
        try:
            landmark_id = self.get_landmark_id(name.upper())
        except Exception as e:
            print(name)
            raise e
        return self.landmarks.landmark[landmark_id]
    
    def get_landmark_location(self, name: str):
        w, h = self.dim
        landmark = self.get_landmark(name)
        return int(landmark.x * w), int(landmark.y * h)
    
    def get_body_orientation(self):
        left_shoulder = self.get_landmark_location('left_shoulder')
        right_shoulder = self.get_landmark_location('right_shoulder')
        left_hip = self.get_landmark_location('left_hip')
        right_hip = self.get_landmark_location('right_hip')
        return get_angle_from_vertical(left_shoulder, left_hip) + get_angle_from_vertical(right_shoulder, right_hip)
    
    
    def add_requirements(self, name: str, target_angle: int):
        if self.mirror:
            if name.startswith('left_'):
                mirror_name = name.replace('left_', 'right_')
            else:
                mirror_name = name.replace('right_', 'left_')
            self.mirror_requirements.append((mirror_name, -target_angle))
        self.pose_requirements.append((name, target_angle))


    def check_single_requirement(self, requirement: tuple) -> bool:
        name, target_angle = requirement
        
        triplet = triplets[name]
        pt, left, right = (self.get_landmark_location(joint) for joint in triplet)
        angle = get_angle_at_joint(pt, left, right)
        
        output = angle_in_range(angle, target_angle, self.angle_margin)
        if self.angles.get(name):
            valid = output or self.angles.get(name)[2]
        else:
            valid = output
        self.angles[name] = [pt, round(angle, 1), valid]
        

        return output
    
    def check_requirements(self):
        self.angles = {}
        if self.mirror:
            pose = all([self.check_single_requirement(req) for req in self.pose_requirements])
            mirror = all([self.check_single_requirement(req) for req in self.mirror_requirements])
            
            return mirror or pose
        else:
            return all([self.check_single_requirement(req) for req in self.pose_requirements])

    
    
    

