from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assignment import PlayerBallAssigner

def main():
    print("Reading video...")
    frames = read_video('input_videos/video.mp4')

    # Initialize Tracker
    print("Detecting and tracking objects...")
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # interpolatew ball positions
    print("Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball(tracks["ball"])

    # Assign Player Teams
    print("Assigning player Teams...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks["players"][0])
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)

            tracks["players"][frame_num][player_id]["team"] = team
            
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    # Assign ball to player
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True

    # Draw output
    print("Drawing annotations...")
    ## Draw tracks
    out_frames = tracker.draw(frames, tracks)
    

    print("Saving output...")
    save_video(out_frames,'output_videos/vid.avi')

    print("Done!!!")

if __name__ == '__main__':
    main()