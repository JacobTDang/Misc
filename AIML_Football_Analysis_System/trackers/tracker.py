# Imports from YOLO object detection model and supervision library
from ultralytics import YOLO
import supervision as sv
import os
import pickle
import sys
import cv2
import numpy as np
sys.path.append('../')
from utils import get_bbox_width, get_center_of_bbox


class Tracker:
    # constructor to create the tracker using some model
    def __init__(self, model_path):
        # Creates a YOLO object detection model
        self.model = YOLO(model_path)
        
        # Creates a ByteTracker to track object, from supervision library
        self.tracker = sv.ByteTrack()
    
    # Method to detect object depending on frames
    def detect_frames(self, frames):
        # Proess 20 frames at a time for good memory management
        batch_size = 20 
        
        # Initialize empty list to store all detections
        detections = []
        
        # Start to process the batch of 20 frames
        for i in range(0,len(frames), batch_size):
            
            # YOLO detection on batch of frames with a confidence threshold of 0.1
            # frames[i:i+batch_size] would select from from i to i + batch_size
            # only confidence score of 0.1 (10%) will be included in the result
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)
            
            # Increment the amount of detections that were detected after batch to the main list
            detections += detections_batch
            
    
        return detections
    
    
    def get_object_tracking(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
                
        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            # Using map and key and inversing them
            class_names_inv = {v:k for k,v in cls_names.items()}

            # Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert Goal Keeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = class_names_inv["player"]

            # Track objects
            detection_withtracks = self.tracker.update_with_detections (detection_supervision)

            # Initialize dictionaries for this frame
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Process tracked detections
            for frame_detection in detection_withtracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == class_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == class_names_inv['referee']:  # Fixed typo in 'referee'
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}  # Fixed key name

            # Process ball detections (not tracked)
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == class_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
            

    def draw_ellipse(self, frame, bbox, color, track_id):
        # bottom-y coordinate (if bbox = [x1, y1, x2, y2])
        y2 = int(bbox[3])

        # <-- call the utility and unpack its return -->
        x_center, _ = get_center_of_bbox(bbox)
        width      = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(int(x_center), y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )
        
        rectangle_width = 40
        rectangle_height = 20
        
        x1_rect = x_center - rectangle_width/2
        x2_rect = x_center + rectangle_width/2
        
        y1_rect = (y2 - rectangle_height/2) + 15
        y2_rect = (y2 + rectangle_height/2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                ((int)(x1_rect), (int)(y1_rect)),
                ((int)(x2_rect), (int)(y2_rect)),
                color,
                cv2.FILLED
            )
            
            x1_text = x1_rect + 12
            
            if track_id > 99:
                x1_text -= 10
                
            cv2.putText(
                frame,
                f"{track_id}",
                ((int)(x1_rect), (int)(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )    
                
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)
        
        triangle_points = np.array(
            [
                [x,y],
                [x-10,y-20],
                [x+10,y-20]
            ]
        )
        
        # Drawing the triangle
        cv2.drawContours(frame,[triangle_points], 0, color, cv2.FILLED)
        
        # Drawing the border for the triangle
        cv2.drawContours(frame,[triangle_points], 0, (0,0,0), 2)
        
        return frame
        


    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            # Draw a circle below user
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            
            # Drawing Players
            for track_id, player in player_dict.items():
                # setting the team color, if not default to red color
                color = player.get("team_colors", (0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], (0,0,255), track_id)
                
            # Drawing Referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255), None)    
                
            # Draw ball
            for track_id, ball in  ball_dict.items():
                frame = self.draw_triangle(frame,ball["bbox"], (255,0,0))
                
            output_video_frames.append(frame)
        
        return output_video_frames