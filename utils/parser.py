__author__ = 'Gohur Ali'
__version__ = 2.0
import cv2              # Opening video and grabbing frames
import os               # Directory searching  
import argparse         # Command line arguement seeting
import time             # Getting current time for naming convention
import numpy as np      # Python float to Numpy float conversion
from tqdm import tqdm   # Progress bar for visualizing number of frames
import imageio          # Create gifs from list of images
"""
Python script for frame extraction from video files.

How to run this script:
In a bash or windows command prompt terminal run the following command
python3 pbar_video_parser.py <video location> <output directory name> <number of frames to skip>

Note: 
0.5 will save a frame every half a second or 2 images a second
5 will save a frame for every 5 seconds in the video

You must have OpenCV and Numpy to be able to run this program as they are dependencies:
    To install OpenCV:
        pip install opencv-python
    To install numpy
        pip install numpy
    To install tqdm
        pip install tqdm

Output images are named YYYYMMDDhourMinute_pictureNum.png
"""

parser = argparse.ArgumentParser(description='Process and Cut Video Frames')
parser.add_argument('--input_video', help='name of the directory that the images will be stored in')
parser.add_argument('--output_dir', help='the video path in filesystem')
parser.add_argument('--seconds_delay',help='number of seconds delayed before frame is saved')
parser.add_argument('--make-gif',action='store_true',help='create gif image video')
parser.add_argument('--frames_path',help='location to image frames to create gif')
parser.add_argument('--output_loc',help='location where gif will be located')
args = parser.parse_args()

def get_current_time():
    '''
    Using time module to get current time and using it 
    as a time convention.
    '''
    minute = time.localtime().tm_min
    hour = time.localtime().tm_hour
    day = time.localtime().tm_mday
    month = time.localtime().tm_mon
    year = time.localtime().tm_year
    return minute, hour, day, month, year

def get_info(args):
    '''
    Returns information about the video capture
    :param: args - arguments taken from command line
    :return: duration of the video capture and total num of frames
    '''
    video_path = args.input_video
    vid_cap = cv2.VideoCapture(video_path)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    frame_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count/fps
    return duration, frame_count

def frame_estimation(args, duration):
    '''
    Calculating the number of frames that will be saved
    to disk in the output directory
    :param: args - arguments taken from command line
    :param: duration - the duration of the video capture
    :return: num of frames to be saved
    '''
    img_per_sec = 1 / float(args.seconds_delay)
    frames_est = img_per_sec * duration
    return frames_est

def parse_all_frames(args, frame_count):
    '''
    If 0 seconds was specified in seconds_delay arg
    then parse_all_frames extracts all frames from 
    the video capture
    :param: args - arguments taken from command line
    :param: frame_count - number of frames in the video capture
    '''
    pbar = tqdm(total=frame_count)
    video_path = args.input_video
    frame_counter = 0
    vid_cap = cv2.VideoCapture(video_path)
    minute, hour, day, month, year = get_current_time()

    if(os.path.exists(args.output_dir) == False):
        os.mkdir(args.output_dir)

    while(vid_cap.isOpened()):
        ret, frame = vid_cap.read()
        if(ret == True):
            im_name = str(year) + str(month) + str(day) + str(hour) + str(minute) + '_' +str(frame_counter) + '.png'
            tqdm.write(('Creating image: ' + im_name))
            save_loc = args.output_dir + '/' + im_name
            cv2.imwrite(save_loc,frame)
            pbar.update(1)
            frame_counter += 1
        if(frame_counter == frame_count):
            break
    pbar.close()
    vid_cap.release()
    cv2.destroyAllWindows()       

def parse_video(args, frame_est):
    '''
    Grabbing particular frames from the video capture and saving them to disk
    at a given location that is taken from the command line.

    :param: Arguments taken from the command line
    :param: The estimated number of frames to be saved to disk
    '''
    video_path = args.input_video
    frame_count = 0
    vid_cap = cv2.VideoCapture(video_path)
    minute, hour, day, month, year = get_current_time()

    pbar = tqdm(total=frame_est, leave=False)

    if(os.path.exists(args.output_dir) == False):
        os.mkdir(args.output_dir)

    while(vid_cap.isOpened()):
        if(float(args.seconds_delay) == 0):
            vid_cap.set(cv2.CAP_PROP_POS_MSEC, frame_count)
        elif(float(args.seconds_delay) > 0):
            frame_skip_rate = np.float32( float(frame_count) * float(args.seconds_delay) * 1000)
            vid_cap.set(cv2.CAP_PROP_POS_MSEC,(frame_skip_rate))

        ret, frame = vid_cap.read()        
        if(ret == True):
            im_name = str(year) + str(month) + str(day) + str(hour) + str(minute) + '_' +str(frame_count) + '.png'
            tqdm.write(('Creating image: ' + im_name))
            save_loc = args.output_dir + '/' + im_name
            cv2.imwrite(save_loc,frame)
            pbar.update(1)
        else:
            break
        frame_count += 1
    pbar.close()
    vid_cap.release()
    cv2.destroyAllWindows()

def make_gif(dir_loc):
    images = []
    for f in os.listdir(dir_loc):
        im = imageio.imread(dir_loc+f)
        images.append(im)
    minute, hour, day, month, year = get_current_time()
    im_name = str(year) + str(month) + str(day) + str(hour) + str(minute) + '.gif'
    imageio.mimsave(args.output_loc+im_name,images)
    print('-- Saved gif at location [' +args.output_loc+'] --' )

def main():
    if(args.make_gif):
        make_gif(args.frames_path)
    else:
        duration, frame_count = get_info(args)
        if(float(args.seconds_delay) == 0):
            parse_all_frames(args, frame_count)
        else:
            frame_est = frame_estimation(args, duration)
            parse_video(args, frame_est)

if __name__ == '__main__':
    main()