import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
from .onnxdet import inference_detector
from .onnxpose import inference_pose

class Wholebody:
    def __init__(self):
        device = 'cuda:0'
        providers = ['CPUExecutionProvider'
                 ] if device == 'cpu' else ['CUDAExecutionProvider']
        onnx_det = 'annotator/ckpts/yolox_l.onnx'
        onnx_pose = 'annotator/ckpts/dw-ll_ucoco_384.onnx'

        self.session_det = ort.InferenceSession(path_or_bytes=onnx_det, providers=providers)
        self.session_pose = ort.InferenceSession(path_or_bytes=onnx_pose, providers=providers)
    
    def __call__(self, oriImg):
        det_result = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[
            ..., :2], keypoints_info[..., 2]
        
        return keypoints, scores

class Wholebody2D:
    def __init__(self, yolo_model='yolov8x.pt', classes=[0], tracker="botsort.yaml", conf=0.1, iou=0.1, persist=True, imgsz=1920, tracked_id=1):
        device = 'cuda:0'
        providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']

        onnx_pose = 'annotator/ckpts/dw-ll_ucoco_384.onnx'
        self.yolo_det = YOLO(yolo_model)
        self.session_pose = ort.InferenceSession(path_or_bytes=onnx_pose, providers=providers)
        self.tracked_id = tracked_id
        self.classes = classes
        self.tracker = tracker
        self.conf = conf
        self.iou = iou
        self.persist = persist
        self.imgsz = imgsz
        self.det_result = None
        self.tracked_succesfull = False
        #self.det_result_old = None

    def __call__(self, oriImg):
        model_output = self.yolo_det.track(oriImg, classes=self.classes, tracker=self.tracker, conf=self.conf, iou=self.iou, persist=self.persist, imgsz=self.imgsz)
        #self.det_result = self.det_result_old

        # Track the detected objects
        for r in model_output:
            boxes = r.boxes
            for box in boxes:
                if box.cls[0] == 0:
                    track_id = box.id
                    if self.tracked_id is not None and track_id == self.tracked_id:
                        self.det_result = box[0, :].data[:, 0:4].cpu().numpy().copy()
                        self.tracked_succesfull = True
                    else:
                        self.tracked_succesfull = False

        #self.det_result_old = self.det_result
        #if self.tracked_succesfull == True:
        self.det_result[0, 3] = self.det_result[0, 3] + 100

        keypoints, scores = inference_pose(self.session_pose, self.det_result, oriImg)

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]

        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[
            ..., :2], keypoints_info[..., 2]
        
        return keypoints, scores
