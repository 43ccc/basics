import cv2
import math
import os

class FrameExtractor():

    def __save_frames(self, video, output_path, save_every_n_frames, name_prefix='frame'):
        os.makedirs(output_path, exist_ok=True)

        frame_count = 0
        success = True

        while success:
            success, frame = video.read()
            # Save every "save_every_n_frames" frames
            if frame_count % save_every_n_frames == 0:
                image_path = f"{output_path}/{name_prefix}_{frame_count}.png"
                cv2.imwrite(image_path, frame)

            frame_count += 1
    
    def save_frames_from_video(self, video_path, output_path, save_every_n_seconds, name_prefix='frame'):
        # Load the video
        video = cv2.VideoCapture(video_path)
        is_open = video.isOpened()

        # Calculate save_every_n_frames
        fps = math.ceil(video.get(cv2.CAP_PROP_FPS))
        save_every_n_frames = fps * save_every_n_seconds

        # Save frames if video has ben successfully opened
        if is_open:
            self.__save_frames(video, output_path, save_every_n_frames, name_prefix)
        
        # Close the video and return whether it was a success or not
        video.release()

        return is_open
    
    def save_frames_from_all_videos(self, video_folder_path, output_path, save_every_n_seconds):
        # Go over all videos in the given video folder and save their frames
        for idx, video in enumerate(os.listdir(video_folder_path)):
            final_output_path = f'{output_path}/{video}'
            video_path = f'{video_folder_path}/{video}'
            name_prefix = f'video_{idx}'
            success = self.save_frames_from_video(video_path, final_output_path, save_every_n_seconds, name_prefix=name_prefix)
            
            # Print out wheter it was a success
            if success:
                print(f'Frames for Video {video} have been saved')
            else:
                print(f'Failed to extract frames for {video}')

