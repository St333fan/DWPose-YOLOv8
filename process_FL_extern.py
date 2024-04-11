from utils import *
import numpy as np
Setup_environment()
from hpe_wrapper import Wrapper_3Dpose

model_3D = '/home/imw/PycharmProjects/ergomaps/VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin'
pose_3d = Wrapper_3Dpose(model_3D)

# Video resolution
resolution = {
    'w': 1280,
    'h': 720,
    'fps': 30
}

metadata = {}
metadata['layout_name'] = 'coco'
metadata['num_joints'] = 17
metadata['keypoints_symmetry'] = [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]
metadata['video_metadata'] = {'data_2d': resolution}

kp = np.load('/home/imw/PycharmProjects/DWPose/ControlNet-v1-1-nightly/stefan_p_1_00000473.npy')

# Create a blank image
image = np.zeros((720, 1280, 3), dtype=np.uint8)

# Iterate through the keypoints and draw them on the image
for i, (x, y, conf) in enumerate(kp[0,:]):
    # Draw the keypoint
    cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), 2)

    # Add the row number as text
    cv2.putText(image, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Display the image
cv2.imshow('Keypoints', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

keypoints = {'data_2d': {'custom': kp}}
print("Finished processing video!")


x = {
    'start_frame': 0,  # Inclusive
    'end_frame': len(keypoints),  # Exclusive
    'bounding_boxes': None,  # boxes
    'keypoints': keypoints,  # keypoints
}

data_3d = pose_3d.predict_3D_poses(x, metadata)
pose_3d.render_video_output(output_path="3D_result.mp4")
