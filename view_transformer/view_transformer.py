import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        court_width = 68
        court_length = 23.32

        self.pixel_vertices = np.array([
            [110,1035],
            [265,275],
            [910,260],
            [1640,915]
        ])

        self.target_vertices = np.array([
            [0,court_width],
            [0,0],
            [court_length,0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, p):
        pt = (int(p[0]), int(p[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, pt, False) >=0
        if not is_inside:
            return None
        
        reshaped_pt = p.reshape(-1,1,2).astype(np.float32)
        transformed_pt = cv2.perspectiveTransform(reshaped_pt, self.perspective_transformer)

        return transformed_pt.reshape(-1,2) 


    def add_transform_to_tracks(self, tracks):
        for obj, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    transformed_pos = self.transform_point(position)

                    if transformed_pos is not None:
                        transformed_pos = transformed_pos.squeeze().tolist()
                    tracks[obj][frame_num][id]['transformed_position'] = transformed_pos