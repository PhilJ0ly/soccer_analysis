import pickle
import cv2
import numpy as np
import sys
import os
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        self.min_dis = 5

        self.lk_params = dict(
            winSize = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_gray)
        mask_features[:,0:20]=1
        mask_features[:,900:1050]=1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # read from stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)

        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_dis = 0
            mvmt_x, mvmt_y = 0, 0

            for i, (new,old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                dis = measure_distance(new_features_point, old_features_point)
                if dis > max_dis:
                    max_dis = dis
                    mvmt_x, mvmt_y = measure_xy_distance(old_features_point, new_features_point)
            
            if max_dis>self.min_dis:
                camera_movement[frame_num] = [mvmt_x, mvmt_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            
            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
                
        return camera_movement

    def adjust_positions_tracks(self, tracks, camera_mvmt_frame):
        for obj, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    mvmt = camera_mvmt_frame[frame_num]
                    position_adjusted = (position[0]-mvmt[0], position[1]-mvmt[1])
                        
                    tracks[obj][frame_num][track_id]['position_adjusted'] = position_adjusted

    def draw_camera_movement(self, frames, camera_mvmt_frame):
        out_frames = []

        for frame_num, frame in enumerate(frames):
            frame =  frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (500,100), (255,255,255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

            x_mvmt, y_mvmt = camera_mvmt_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x_mvmt:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_mvmt:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

            out_frames.append(frame)

        return out_frames