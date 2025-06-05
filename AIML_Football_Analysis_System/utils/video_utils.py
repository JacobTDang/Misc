import cv2

# function to capture video frames
def read_video(video_path):
    # use a video capture object that opens the video files based on the video file path
    cap = cv2.VideoCapture(video_path)
    # Initialie an empty list to hold the video frames
    frames = []
    
    # while true...
    while True:
        # cap.read() returns two values:
        # ret: a boolean value. True if frame was successfully read, false otherwise
        # frame is the actual image data; frame is a numPy variable that receives a NumPy array that is 3D (H,W,channels(BRG color values))
        ret, frame = cap.read()
        
        # if ret is false / frame is not read then...
        if not ret:
            break
        
        # else if we got a valid frame, then add it to the list
        frames.append(frame)
        
    # return the complete list of frames    
    return frames

# Function used to save video with parameters
# number of frames and video file path
def save_video(output_video_frames, output_video_path):
    
    # four character code that ientifies the video codec
    fourcc  = cv2.VideoWriter_fourcc(*"XVID")
    
    # Creating the video writer object paramters
    # Path to save video
    # Codec (XVID)
    # The frames rate (24 frames per second)
    # Resolution (width, height) that is based of the first frame
    out = cv2.VideoWriter(output_video_path, fourcc, 24, 
                          (output_video_frames[0].shape[1], # x coordinates
                           output_video_frames[0].shape[0]) # y coordinates
                          )
    
    # loop through each frame in the list and write it to the video file
    for frame in output_video_frames:
        out.write(frame)
    
    # close the video file and release resources    
    out.release()