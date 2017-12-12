import os, cv2

def make_video(image_folder, video_name):
    num_frames = len(os.listdir(image_folder))
    frame = cv2.imread(os.path.join(image_folder, '0.jpg'))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, -1, 1, (width,height))
    for k in range(num_frames):
        video.write(cv2.imread(os.path.join(image_folder, str(k)+'.jpg')))
        
    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    make_video('./dump_train', 'video.avi')