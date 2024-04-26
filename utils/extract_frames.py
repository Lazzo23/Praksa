import cv2

def extract_frames(video_path, start_time, end_time, num_frames):

    """ 
    Function extract_frames() reads video from "video_path" and crops it
    to the subclip [start_time : end_time], where both times are in seconds.
    Then function crops each frame to 480x350 image and appends it to the 
    list "frames". Frames are evenly extracted through the whole subclip
    based on number of returned frames, which is stored in "num_frames".

    extract_frames(string, int, int, int) -> list of numpy.ndarrays
    
    """

    # Open video
    video = cv2.VideoCapture(video_path) 
    fps = video.get(cv2.CAP_PROP_FPS)

    # Check if we can get desired number of frames
    total_frames = (end_time - start_time) * fps
    if num_frames < 1 or end_time <= start_time or total_frames < num_frames:
        return []
    
    # Calculate number of skip frames
    skip_frames = total_frames // num_frames
    
    # Set the start and end frame
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # List of extracted frames
    frames = []
    
    # Read frame by frame
    current_frame = start_frame
    while current_frame < end_frame:
        
        # Read next frame
        ret, frame = video.read()

        # Break if reading is unsuccessfull or 
        # if we have desired number of frames
        if not ret or len(frames) == num_frames:
            break

        # Crop each frame
        frame = frame[0:350, 0:480]
        
        # Add cropped frame to list
        frames.append(frame)
        
        # Skip frames
        current_frame += skip_frames

        # Set new frame
        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    
    # Release video
    video.release()
    
    return frames

video_path = "video.mp4"
start_time = 10  
end_time = 20  

frames = extract_frames(video_path, start_time, end_time, 10)
for i, f in enumerate(frames):
    cv2.imwrite(f"frames/{i}.png", f)