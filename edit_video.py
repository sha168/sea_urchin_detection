import moviepy.editor as mp

path_in = "/Users/sha168/Library/CloudStorage/OneDrive-UiTOffice365/video_BYEDP190675_2023-02-20_123159.MP4"
path_out = "/Users/sha168/Library/CloudStorage/OneDrive-UiTOffice365/video_BYEDP190675_2023-02-20_123159_resized.MP4"
clip = mp.VideoFileClip(path_in)
clip_resized = clip.resize(height=512)  # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
clip_resized.write_videofile(path_out)


# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# ffmpeg_extract_subclip("video1.mp4", start_time, end_time, targetname="test.mp4")