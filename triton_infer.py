import argparse
import os
import sys
import time

from sklearn.decomposition import PCA
import cv2
import numpy as np
import sklearn
import tritonclient.grpc as grpcclient

print("------- ", sys.path)
sys.path.append(os.path.abspath('..'))
print("------- ", sys.path)
import config.parameter as parameter
import config.system


class BaseTriton():
    def __init__(self, model_name='', input_name=[], input_type=[], input_dim=[], output_name=[], \
                url=config.system.URL_SERVER_AI, verbose=False, ssl=False, root_certificates=None, private_key=None, \
                certificate_chain=None, client_timeout=None, grpc_compression_algorithm=None, static=False):
        self.triton_client = grpcclient.InferenceServerClient(
          url=url,
          verbose=verbose,
          ssl=ssl,
          root_certificates=root_certificates,
          private_key=private_key,
          certificate_chain=certificate_chain
        )

        self.model_name = model_name
        self.input_name = input_name  # 'data'
        self.input_type = input_type  # "FP32"
        self.input_dim = input_dim  # [1, 3, 112, 112]
        self.output_name = output_name
        self.client_timeout = client_timeout
        self.static = static

    def predict(self, inputs_data):
        # Infer
        inputs = []
        outputs = []
        for i, input_data in enumerate(inputs_data):
            inputs.append(grpcclient.InferInput(self.input_name[i], input_data.shape, self.input_type[i]))
            # input0_data = self.preprocess(img)

            # Initialize the data
            inputs[i].set_data_from_numpy(input_data)

        for i, out_name in enumerate(self.output_name):
            outputs.append(grpcclient.InferRequestedOutput(out_name))

        # Test with outputs
        results = self.triton_client.infer(
          model_name=self.model_name,
          inputs=inputs,
          outputs=outputs,
          client_timeout=self.client_timeout,
          headers={'test': '1'}
        )

        if self.static:
            statistics = self.triton_client.get_inference_statistics(model_name=self.model_name)
            # print(statistics)
            if len(statistics.model_stats) != 1:
                print("FAILED: Inference Statistics")
                sys.exit(1)

        # Get the output arrays from the results
        outputs_data = [results.as_numpy(self.output_name[i]) for i in range(len(self.output_name))]
        return outputs_data


class ArcfaceTriton(BaseTriton):
    def __init__(self, model_name=parameter.ARC_MODEL_NAME, input_name=parameter.ARC_INPUT_NAME, input_type=parameter.ARC_INPUT_TYPE, \
                input_dim=parameter.ARC_INPUT_DIM, output_name=parameter.ARC_OUTPUT_NAME):
        super().__init__(model_name=model_name, input_name=input_name, input_type=input_type, input_dim=input_dim, output_name=output_name, \
                url=config.system.URL_SERVER_AI2, verbose=False, ssl=False, root_certificates=None, private_key=None, \
                certificate_chain=None, client_timeout=None, grpc_compression_algorithm=None, static=False)

    def preprocess(self, img):
        if len(img.shape) == 3:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0).astype(np.float32)
        else:
            img = img[:, :, :, ::-1]
            img = np.transpose(img, (0, 3, 1, 2)).astype(np.float32)
        return img

    def get(self, img, norm=True):
        # Infer
        input0_data = self.preprocess(img)

        outputs_data = self.predict([input0_data])
        if norm:
            embedding = sklearn.preprocessing.normalize(outputs_data[0]).flatten()
        return embedding


class PredictTriton5pointv2(BaseTriton):
    def __init__(self, model_name=parameter.MNETV2_MODEL_NAME, input_name=parameter.MNETV2_INPUT_NAME, input_type=parameter.MNETV2_INPUT_TYPE, \
                input_dim=parameter.MNETV2_INPUT_DIM, output_name=parameter.MNETV2_OUTPUT_NAME):
        super().__init__(model_name=model_name, input_name=input_name, input_type=input_type, input_dim=input_dim, output_name=output_name, \
                url=config.system.URL_SERVER_AI2, verbose=False, ssl=False, root_certificates=None, private_key=None, \
                certificate_chain=None, client_timeout=None, grpc_compression_algorithm=None, static=False)


class PredictTriton5point(BaseTriton):
    def __init__(self, model_name=parameter.MNET_MODEL_NAME, input_name=parameter.MNETV2_INPUT_NAME, input_type=parameter.MNETV2_INPUT_TYPE, \
                input_dim=parameter.MNETV2_INPUT_DIM, output_name=parameter.MNETV2_OUTPUT_NAME):
        super().__init__(model_name=model_name, input_name=input_name, input_type=input_type, input_dim=input_dim, output_name=output_name, \
                url=config.system.URL_SERVER_AI2, verbose=False, ssl=False, root_certificates=None, private_key=None, \
                certificate_chain=None, client_timeout=None, grpc_compression_algorithm=None, static=False)


class PredictTriton106point(BaseTriton):
    def __init__(self, model_name=parameter.DET_MODEL_NAME, input_name=parameter.DET_INPUT_NAME, input_type=parameter.DET_INPUT_TYPE, \
                input_dim=parameter.DET_INPUT_DIM, output_name=parameter.DET_OUTPUT_NAME):
        super().__init__(model_name=model_name, input_name=input_name, input_type=input_type, input_dim=input_dim, output_name=output_name, \
                url=config.system.URL_SERVER_AI2, verbose=False, ssl=False, root_certificates=None, private_key=None, \
                certificate_chain=None, client_timeout=None, grpc_compression_algorithm=None, static=False)
