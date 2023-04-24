import moviepy.editor as mp
import glob

path_in = '/Users/sha168/Downloads/videos_test'
path_out = '/Users/sha168/Downloads/videos_preprocessed'
videos = glob.glob(path_in + '/*.MP4')

# path_in = "/Users/sha168/Library/CloudStorage/OneDrive-UiTOffice365/video_BYEDP190675_2023-02-20_123159.MP4"
# path_out = "/Users/sha168/Downloads/video_BYEDP190675_2023-02-20_123159_preprocessed.MP4"

for video in videos:

    file_name = video.split(path_in)[-1]

    clip = mp.VideoFileClip(video)
    duration = clip.duration
    max_length = 120
    if duration > max_length:
        clip = clip.subclip((duration/2)-(max_length/2), (duration/2)+(max_length/2))
    clip_resized = clip.resize(height=512)  # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
    clip_resized.write_videofile(path_out + file_name)



# # from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# # ffmpeg_extract_subclip("video1.mp4", start_time, end_time, targetname="test.mp4")
# import cv2
# import glob
# count = 0
# frame = 0
#
# videos = glob.glob('/Users/sha168/Library/CloudStorage/OneDrive-UiTOffice365/Videoer/*.mp4') + glob.glob('/Users/sha168/Library/CloudStorage/OneDrive-UiTOffice365/Videoer/*.MP4')
#
# for i, video in enumerate(videos):
#     print(i+1, '/', len(videos), ' : ', video)
#     try:
#         print('read video')
#         video = video[:-3] + 'MP4'
#         vidcap = cv2.VideoCapture(video)
#         success, image = vidcap.read()
#     except:
#         print('moving to next video')
#         continue
#
#     while success:
#         if count % 200 == 0:
#             cv2.imwrite("/Users/sha168/Downloads/video_frames/frame%d.jpg" % frame, image)     # save frame as JPEG file
#             frame += 1
#             print('Read a new frame: ', count, success)
#
#         success,image = vidcap.read()
#         count += 1

# import shutil
# files = glob.glob('/Users/sha168/Downloads/video_frames/*')
# files_ = [file[:-4].split('frames/frame')[-1]for file in files]
# files_ = sorted(files_, key=lambda x: float(x))
#
#
# for i, file in enumerate(files_):
#     if i % 2 == 0:
#         src = '/Users/sha168/Downloads/video_frames/frame' + file + '.jpg'
#         dst = '/Users/sha168/Downloads/video_frames_half/frame' + file + '.jpg'
#         shutil.copyfile(src, dst)
