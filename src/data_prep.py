import cv2
import os

'''
This program devides each video from dataset videos into single frames.
'''


videos_path = '../dane/filmy/UCF-101'
videos_files = os.listdir(videos_path)


for i in videos_files:
    frames_action_path = (f'../dane/klatki/{i}')
    if not i.startswith('.'): 
        videos = os.listdir('{}/{}'.format(videos_path, i))
        for movie in videos:
            capture = cv2.VideoCapture('{}/{}/{}'.format(videos_path, i, movie))
            frame_nr = 0
            while True:
                success, frame = capture.read()
                if success: 
                    if not os.path.exists(frames_action_path):
                        os.mkdir(frames_action_path)
                    if not os.path.exists(frames_action_path+f'/{movie}'):
                        os.mkdir(frames_action_path+f'/{movie}')
                    cv2.imwrite(f'{frames_action_path+f"/{movie}"}/frame_{frame_nr}.png'.format(frames_action_path, movie, frame_nr), frame)
                else:
                    break
                frame_nr = frame_nr + 1
            capture.release()