# main.py
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner

def main():
    # 1. Read & track
    video_frames = read_video("input_video/test_265.mp4")
    tracker = Tracker("models/best.pt")
    tracks  = tracker.get_object_tracking(
                  video_frames,
                  read_from_stub=True,
                  stub_path='stubs/track_stubs.pk1'
              )

    # 2. Build teams on first frame
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(
        video_frames[0],
        tracks['players'][0]
    )

    # 3. For every frame & every player, attach team ID + color
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_teams(
                video_frames[frame_num],
                track['bbox'],
                player_id
            )
            track['team']       = team
            track['team_color'] = team_assigner.team_colors[team]

    # 4. Draw & save
    output_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_frames, "output_videos/output_video.avi")


if __name__ == '__main__':
    main()
