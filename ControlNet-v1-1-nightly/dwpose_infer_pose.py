from annotator.dwpose import DWposeDetector2D
import cv2
import numpy as np
c = 1

if __name__ == "__main__":
    pose = DWposeDetector2D(yolo_model='yolov8l.pt', imgsz=1280, tracked_id=1)


    # Open the video file
    video_path = '/home/imw-mmi/Documents/pilotfabrik-dataset/david_p_1/david_p_1_00000152.mp4'
    np_path = 'david_p_1_00000152.npy'

    cap = cv2.VideoCapture(video_path)

    # Get the frame width, height, and FPS
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    output_file = 'output_video.mp4'
    frame_size = (frame_width, frame_height)  # Width, Height
    out_vid = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    keypoints_all = []

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            key_mmpose = np.ones((17, 3))
            oriImg = frame
            out, all, pose_body = pose(oriImg)
            pose_body = all['bodies']['candidate']

            # openpose to mmpose
            key_mmpose[0, :2] = pose_body[0, :]
            key_mmpose[1, :2] = pose_body[15,:]
            key_mmpose[2, :2] = pose_body[14, :]
            key_mmpose[3, :2] = pose_body[17, :]
            key_mmpose[4, :2] = pose_body[16, :]
            key_mmpose[5, :2] = pose_body[5, :]
            key_mmpose[6, :2] = pose_body[2, :]
            key_mmpose[7, :2] = pose_body[6, :]
            key_mmpose[8, :2] = pose_body[3, :]
            key_mmpose[9, :2] = pose_body[7, :]
            key_mmpose[10, :2] = pose_body[4, :]
            key_mmpose[11, :2] = pose_body[11, :]
            key_mmpose[12, :2] = pose_body[8, :]
            key_mmpose[13, :2] = pose_body[12, :]
            key_mmpose[14, :2] = pose_body[9, :]
            key_mmpose[15, :2] = pose_body[13, :]
            key_mmpose[16, :2] = pose_body[10, :]

            key_mmpose[:, 0] *= frame_width
            key_mmpose[:, 1] *= frame_height
            keypoints_all.append(key_mmpose)

            # Convert BGR image to RGB (matplotlib uses RGB)
            image_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            # Write the frame to the output video
            out_vid.write(image_rgb)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            c += 1
            if c > 10 and False:
                break

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

    kp = np.array(keypoints_all, dtype=np.float32).squeeze()
    np.save(np_path, kp)  # save




