# trich xuat dac trung voi pre-train cua repo goc trong insightface
import argparse
import logging
import os
import sys
import time

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import skimage
from easydict import EasyDict as edict

# sys.path.append(os.path.abspath(os.path.join('..', 'align')))
# import align_triton
# sys.path.append(os.path.abspath(os.path.join('..', 'optimizer')))
# from tritonpredict import PredictTritonArcface
sys.path.append(os.path.abspath('..'))
import config.path
from src.detector import face_alignment
from src.extractor import extraction

error_logger = logging.getLogger(config.path.ERROR_LOGGER)
info_logger = logging.getLogger(config.path.INFO_LOGGER)


class Processor:
    def __init__(self):
        # self.args = edict()
        # self.args.image_size = '112,112'
        # self.args.model = '/mnt/NhatNT/f19-ai-cv-face-search/feature_mxnet/models/model-r100-ii/model,0'
        # self.args.gpu = 0
        # self.model = face_model.FaceModel(self.args)

        self.aligner = face_alignment.AlignFaceOpt()
        self.extractor = extraction.Extractor()

    def get(self, img, isAligned=False, mask=False):
        # try:
        if not isAligned:
            aligned, S_face, landmarks_ = self.aligner.alignface(img)

            # print(len(aligned))
            if len(aligned) == 0:
                return [], [], []

            aligned_, rotate = aligned[0]
            if rotate == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                aligned, S_face, landmarks_ = self.aligner.alignface(img)
                # aligned, _ = aligned[0]
            elif rotate == -90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                aligned, S_face, landmarks_ = self.aligner.alignface(img)
                # aligned, _ = aligned[0]
            elif rotate == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
                aligned, S_face, landmarks_ = self.aligner.alignface(img)
                # aligned, _ = aligned[0]
        else:
            aligned = [(img[:, :, ::-1], 0)]
            S_face = []
            faces_pad = []

        faces_align = []
        faces_emb = []
        for i, face_aligned in enumerate(aligned):
            aligned, rotate = face_aligned
            features = self.extractor.get(aligned)  # rgb
            faces_align.append(aligned[:, :, ::-1])  # bgr
            faces_emb.append(features)

        return faces_align, faces_emb, S_face

        # except Exception as e:
        #     error_logger.error("----- error process: %s"%(str(e)))
        #     print("Error: ", e)


# class Extractor:
#     def __init__(self):
#         self.aligner = face_alignment.AlignFaceOpt()
#         self.extractor = extraction.Extractor()

#     def get_emb(self, img):
#         emb_face_norm = model.get_feature(img[:,:,::-1], True)
#         return emb_face_norm

#     def get(self, img, isAligned=False, mask=False):
#         # try:
#         if not isAligned:
#             aligned, S_face, landmarks_, faces_pad = self.aligner.alignface(img)
#             aligned_, rotate = aligned[0]

#             if rotate == 90:
#                 img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
#                 aligned, S_face, landmarks_, faces_pad = self.aligner.alignface(img)
#                 # aligned, _ = aligned[0]
#             elif rotate == -90:
#                 img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#                 aligned, S_face, landmarks_, faces_pad = self.aligner.alignface(img)
#                 # aligned, _ = aligned[0]
#             elif rotate == 180:
#                 img = cv2.rotate(img, cv2.ROTATE_180)
#                 aligned, S_face, landmarks_, faces_pad = self.aligner.alignface(img)
#                 # aligned, _ = aligned[0]
#         else:
#             aligned = [(img[:,:,::-1], 0)]
#             S_face = []
#             faces_pad = []

#         embs = []
#         face_aligns = []
#         masks = []
#         face_pads = []
#         print("aligned: ", len(aligned))
#         for i, face_aligned in enumerate(aligned):
#             emb_face_norm = self.model.get_feature(face_aligned[0], True)
#             embs.append(emb_face_norm)
#             face_aligns.append(face_aligned[0][:,:,::-1])
#             if not isAligned:
#                 face_pads.append(faces_pad[i])
#             if mask:
#                 try:
#                     masks.append(self.aligner.get_landmarksmask(faces_pad[i]))
#                 except:
#                     masks.append(0.4)

#         if mask:
#             return face_aligns, embs, S_face, face_pads, masks
#         else:
#             return face_aligns, embs, S_face, face_pads

#         # except Exception as e:
#         #     print("Error: ", e)

#         # except Exception as e:
#         #     print("Error: ", e)
