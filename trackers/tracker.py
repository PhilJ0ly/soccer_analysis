from ultralytics import YOLO
import supervision  as sv
import cv2

import pickle
import numpy as np
import pandas as pd
import os
import sys

sys.path.append('../')
from utils import get_center_bbox, get_bbox_width, get_foot

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for obj, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if obj == 'ball':
                        position = get_center_bbox(bbox)
                    else:
                        position = get_foot(bbox)
                        
                    tracks[obj][frame_num][track_id]['position'] = position


    def interpolate_ball(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox', []) for x in ball_positions]
        ball_df = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        ball_df = ball_df.interpolate()
        ball_df = ball_df.bfill()

        ball_positions = [{1: {"bbox":x}} for x in ball_df.to_numpy().tolist()]
        return ball_positions
    
    def detect_frames(self, frames):
        batch_sz = 20
        detections = []
        for i in range(0,len(frames), batch_sz):
            detections_batch = self.model.predict(frames[i:i+batch_sz],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
                return tracks
            
        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "refs":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_sv = sv.Detections.from_ultralytics(detection)

            # Convert Goalkeeper to player
            for object_ind, class_id in enumerate(detection_sv.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_sv.class_id[object_ind] = cls_names_inv["player"]

            # Track objects
            detection_tracks = self.tracker.update_with_detections(detection_sv)

            tracks["players"].append({})
            tracks["refs"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv["referee"]:
                    tracks["refs"][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_sv:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            rec_w = 40
            rec_h = 20
            rec_x1 = x_center-rec_w//2
            rec_x2 = x_center+rec_w//2
            rec_y1 = (y2-rec_h//2)+15
            rec_y2 = (y2+rec_h//2)+15

            cv2.rectangle(
                frame, 
                (int(rec_x1),int(rec_y1)), 
                (int(rec_x2),int(rec_y2)), 
                color, 
                cv2.FILLED
            )
            
            txt_x1 = rec_x1+12

            if track_id > 99:
                txt_x1 -= 10
            
            cv2.putText(
                frame,
                f'{track_id}',
                (int(txt_x1), int(rec_y1+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_bbox(bbox)

        tri_pts = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])

        cv2.drawContours(frame, [tri_pts], 0, color, -1)
        cv2.drawContours(frame, [tri_pts], 0, (0,0,0), 2)

        return frame

    def draw_ball_control(self, frame, frame_num, ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350,850),(1900,970),(255,255,255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

        team_control_now = ball_control[:frame_num+1]

        team_1_control = team_control_now[team_control_now==1].shape[0]
        team_1_perc = team_1_control/(frame_num+1)*100
        team_2_perc = 100-team_1_perc

        cv2.putText(frame, f"Team 1 Ball Control: {team_1_perc:.2f}%", (1400,900),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2_perc:.2f}%", (1400,950),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame
    
    def draw(self, frames, tracks, ball_control):
        out_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ref_dict = tracks["refs"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0,0,255))

            # Draw refs
            for track_id, ref in ref_dict.items():
                frame = self.draw_ellipse(frame, ref["bbox"], (0,255,255))

            # Draw ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))

            # Draw Team Ball Control
            frame = self.draw_ball_control(frame, frame_num, ball_control)

            out_frames.append(frame)
        
        return out_frames

        