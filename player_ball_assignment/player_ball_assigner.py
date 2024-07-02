import sys
sys.path.append('../')
from utils import get_center_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_distance = 70

    def assign_ball_player(self, players, ball_bbox):
        ball = get_center_bbox(ball_bbox)

        min_dis = 999
        assigned_player = -1

        for id, player in players.items():
            player_bbox = player['bbox']

            dis_left = measure_distance((player_bbox[0],player_bbox[-1]), ball)
            dis_right = measure_distance((player_bbox[2],player_bbox[-1]), ball)
            dis = min(dis_left, dis_right)

            if dis < self.max_distance:
                if dis < min_dis:
                    min_dis = dis
                    assigned_player = id

        return assigned_player
