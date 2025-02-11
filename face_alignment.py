import datetime
import glob
import logging
import os
import sys
from time import time

import cv2
import numpy as np
import skimage.transform as trans

# import face_recognition
from .landmarks106 import Handler_TRT

sys.path.append(os.path.abspath('..'))
import config.path
from src.detector.face_detection import FaceDetector, FaceDetectorv2
from src.utils import square_crop

error_logger = logging.getLogger(config.path.ERROR_LOGGER)
info_logger = logging.getLogger(config.path.INFO_LOGGER)


class AlignFaceOpt:
    def __init__(self, gpuid=0, gpuid_=1):
        self.detector = FaceDetector()
        self.detector.prepare()
        self.handler = Handler_TRT(det_size=640)

    def draw_landmark5(self, img, faces, landmarks_):
        if faces is not None:
            print('find', faces.shape[0], 'faces')
            for i in range(faces.shape[0]):
                #print('score', faces[i][4])
                box = faces[i].astype(np.int)
                #color = (255,0,0)
                color = (0, 0, 255)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                if landmarks_ is not None:
                    landmark5 = landmarks_[i].astype(np.int)
                    #print(landmark.shape)
                    for l in range(landmark5.shape[0]):
                        color = (0, 0, 255)
                        if l == 0 or l == 3:
                            color = (0, 255, 0)
                        cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

            filename = './detector_test.jpg'
            print('writing', filename)
            cv2.imwrite(filename, img)

    def draw_landmark106(self, img, preds):
        tim = img.copy()
        color = (200, 160, 75)
        for pred in preds:
            pred = np.round(pred).astype(np.int)
            for i in range(pred.shape[0]):
                p = tuple(pred[i])
                cv2.circle(tim, p, 1, color, 1, cv2.LINE_AA)
        return tim

    def get_rotate(self, x1, y1, x2, y2):
        x_abs = abs(x2 - x1)
        y_abs = abs(y2 - y1)
        if max(x_abs, y_abs) < 8:
            return -1
        if x_abs < y_abs:
            if y1 > y2:
                return -90
            else:
                return 90
        else:
            if x2 > x1:
                return 0
            else:
                return 180

    def check_bounding_box(self, x, y, w, h, height, width):
        y1 = y - 1.5*h
        y2 = y + 2*h
        x1 = x - 1.1*w
        x2 = x + 2.1*w
        if y1 < 0:
            y1 = 0
        if y2 > height:
            y2 = height
        if x1 < 0:
            x1 = 0
        if x1 > width:
            x1 = width
        return int(x1), int(x2), int(y1), int(y2)

    def transform_face(self, img, src):
        #output size =112x112
        dst = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655],
                        [70.7299, 92.2041]],
                       dtype=np.float32)

        transform = trans.SimilarityTransform()
        transform.estimate(src, dst)

        # print(src.dot(dst.T))
        M = transform.params[0:2, :]

        face = img[:, :, ::-1]
        face = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return face

    def get_landmarks5(self, img, scale=1.0):
        det_im, det_scale = square_crop(img, 640)
        # print("get landmark5")
        faces, landmarks_ = self.detector.detect(det_im, threshold=0.3)
        # print(faces)
        # self.draw_landmark5(det_im, faces, landmarks_)

        resultFace = []
        S_face = []
        faces_pad = []
        for i, landmarks in enumerate(landmarks_):
            conf = faces[i][4]
            if conf > 0.5:
                left_eye = landmarks[0]
                right_eye = landmarks[1]

                nostril = landmarks[2]
                left_lip = landmarks[3]
                right_lip = landmarks[4]

                # print("location: ", left_eye, right_eye, nostril, left_lip, right_lip)
                src = np.array([left_eye, right_eye, nostril, left_lip, right_lip], dtype=np.float32)

                ## xac dinh mat bi doc de xoay
                x1, y1 = left_eye
                x2, y2 = right_eye
                x3, y3 = right_lip
                x4, y4 = left_lip
                rotate = self.get_rotate(x1, y1, x2, y2)

                arr = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                roi_face = np.array(arr, dtype=np.int32)
                rect = cv2.boundingRect(roi_face)
                x, y, w, h = rect
                # height, width, _ = img.shape
                # x1, x2, y1, y2 = self.check_bounding_box(x, y, w, h, height, width)

                S_face.append(w * h)

                face = self.transform_face(det_im, src)
                # face_pad = img[y1:y2, x1:x2]

                resultFace.append((face, rotate))
                # faces_pad.append(face_pad)
        return resultFace, S_face, landmarks_

    def get_landmarks106(self, img):
        #print("====== landmarks 106 =======")
        landmarks_ = []
        resultFace = []
        S_face = []
        faces_pad = []
        # im = cv2.imread('/home/nhatnt/Pictures/check_error_align/photo_2021-02-08_14-19-04.jpg')
        tim = img.copy()
        preds = self.handler.get(img, get_all=True)
        # color = (200, 160, 75)

        for pred in preds:
            left_eye = pred[38]
            right_eye = pred[88]

            nostril = pred[86]
            left_lip = pred[52]
            right_lip = pred[61]

            # print("location: ", left_eye, right_eye, nostril, left_lip, right_lip)
            src = np.array([left_eye, right_eye, nostril, left_lip, right_lip], dtype=np.float32)

            ## xac dinh mat bi doc de xoay
            x1, y1 = left_eye
            x2, y2 = right_eye
            x3, y3 = right_lip
            x4, y4 = left_lip
            rotate = self.get_rotate(x1, y1, x2, y2)

            arr = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            roi_face = np.array(arr, dtype=np.int32)
            rect = cv2.boundingRect(roi_face)
            x, y, w, h = rect
            # height, width, _ = img.shape
            # x1, x2, y1, y2 = self.check_bounding_box(x, y, w, h, height, width)

            S_face.append(w * h)
            face = self.transform_face(img, src)
            # face_pad = img[y1:y2, x1:x2]
            landmarks_.append((left_eye, right_eye, nostril, left_lip, right_lip))
            resultFace.append((face, rotate))
            # faces_pad.append(face_pad)
        landmarks_ = np.asarray(landmarks_)
        return resultFace, S_face, landmarks_

    def alignface(self, img):
        # # t1 = time()
        # resultFace, S_face, landmarks_ = self.get_landmarks106(img)
        # #print("Time 106: ", time() - t1)
        # if landmarks_.shape[0] > 0:
        #     return resultFace, S_face, landmarks_
        # else:
        #     # t2 = time()
        #     # print("landmark 5")
        #     resultFace, S_face, landmarks_ = self.get_landmarks5(img)
        #     #print("Time 5: ", time() - t2)
        #     return resultFace, S_face, landmarks_
        resultFace, S_face, landmarks_ = self.get_landmarks5(img)
        return resultFace, S_face, landmarks_

    def align5(self, img):
        # try:
        resultFace, S_face, landmarks_ = self.get_landmarks5(img)
        return resultFace, S_face, landmarks_
        # except Exception as e:
        #     resultFace, S_face, landmarks_ = [], [], []
        #     return resultFace, S_face, landmarks_
        #     resultFace, S_face, landmarks_ = [], [], []
        #     return resultFace, S_face, landmarks_
