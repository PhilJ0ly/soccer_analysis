import sys
import cv2
sys.path.append('../')
from utils import measure_distance, get_foot

class SpeedDisEstimator():
    def __init__(self):
        self.frame_window = 5
        self.fps = 24

    def speed_dis_to_tracks(self, tracks):
        total_dis = {}

        for obj, obj_tracks in tracks.items():
            if obj == "ball" or obj == "referees":
                continue
            num_of_frames = len(obj_tracks)

            for frame_num in range(0, num_of_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window, num_of_frames-1)

                for id, _ in obj_tracks[frame_num].items():
                    if id not in obj_tracks[last_frame]:
                        continue

                    start_pos = obj_tracks[frame_num][id]['transformed_position']
                    end_pos = obj_tracks[last_frame][id]['transformed_position']

                    if start_pos is None or end_pos is None:
                        continue

                    dis_covered = measure_distance(start_pos, end_pos)
                    time_elapsed = (last_frame - frame_num)/self.fps
                    speed = dis_covered/time_elapsed * 3.6

                    if obj not in total_dis:
                        total_dis[obj] = {}
                    
                    if id not in total_dis[obj]:
                        total_dis[obj][id] = 0

                    total_dis[obj][id] += dis_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if id not in tracks[obj][frame_num_batch]:
                            continue

                        tracks[obj][frame_num_batch][id]['speed'] = speed
                        tracks[obj][frame_num_batch][id]['distance'] = total_dis[obj][id]

    def draw_speed_dis(self, frames, tracks):
        out_frames = []
        for frame_num, frame in enumerate(frames):
            for obj, obj_tracks in tracks.items():
                if obj == "ball" or obj == "referees":
                    continue
                for _, track_info in obj_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get("speed", None)
                        dis = track_info.get('distance', None)

                        if speed is None or dis is None:
                            continue

                        bbox = track_info['bbox']
                        pos = get_foot(bbox)
                        pos = list(pos)
                        pos[1]+=40

                        pos = tuple(map(int,pos))
                        cv2.putText(frame, f"{speed:.2f}km/h", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                        cv2.putText(frame, f"{dis:.2f}m", (pos[0], pos[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            
            out_frames.append(frame)

        return out_frames