EMBEDDING_FEATURE_SIZE = 512
THRESHOLD_FACE = 0.68
TOP_K = 10

## ARCFACE
ARC_MODEL_NAME = 'arcface'
ARC_INPUT_NAME = ['data']
ARC_INPUT_TYPE = ['FP32']
ARC_INPUT_DIM = [[1, 3, 112, 112]]
ARC_OUTPUT_NAME = ['fc1']

## MNET25V2
MNETV2_MODEL_NAME = 'mnet25v2'
MNETV2_INPUT_NAME = ['data']
MNETV2_INPUT_TYPE = ['FP32']
MNETV2_INPUT_DIM = [[1, 3, 640, 640]]
MNETV2_OUTPUT_NAME = ['face_rpn_cls_prob_reshape_stride32', 'face_rpn_bbox_pred_stride32', 'face_rpn_landmark_pred_stride32', 'face_rpn_cls_prob_reshape_stride16', \
                    'face_rpn_bbox_pred_stride16', 'face_rpn_landmark_pred_stride16', 'face_rpn_cls_prob_reshape_stride8', 'face_rpn_bbox_pred_stride8', 'face_rpn_landmark_pred_stride8']

## MNET25
MNET_MODEL_NAME = 'mnet25'
MNET_INPUT_NAME = ['data']
MNET_INPUT_TYPE = ['FP32']
MNET_INPUT_DIM = [[1, 3, 640, 640]]
MNET_OUTPUT_NAME = ['face_rpn_cls_prob_reshape_stride32', 'face_rpn_bbox_pred_stride32', 'face_rpn_landmark_pred_stride32', 'face_rpn_cls_prob_reshape_stride16', \
                    'face_rpn_bbox_pred_stride16', 'face_rpn_landmark_pred_stride16', 'face_rpn_cls_prob_reshape_stride8', 'face_rpn_bbox_pred_stride8', 'face_rpn_landmark_pred_stride8']

## 2D106DET
DET_MODEL_NAME = '2d106det'
DET_INPUT_NAME = ['data']
DET_INPUT_TYPE = ['FP32']
DET_INPUT_DIM = [[1, 3, 192, 192]]
DET_OUTPUT_NAME = ['fc1']
