import os.path as osp
import numpy as np
import onnx
import onnxruntime as ort
import torch
from model.model import Audio2Face
# torch --> onnx

batch_size = 100
test_arr = torch.randn(batch_size,32,64,1).type(torch.FloatTensor) 
dummy_input = torch.tensor(test_arr)
model = Audio2Face(24)
# state_dict = torch.load('/home/ubuntu/Audio2Face/output_mine_head_lr_1e-4_gamma_0.99_bs_128_head_head/checkpoint/Audio2Face/model_lr_1e-4_gamma_0.99_bs_128_head_400.pth', map_location=torch.device('cpu'))
# state_dict = torch.load('/home/ubuntu/Audio2Face/output_4_16_mine_finetune_lr1e-4_gamma_99_bs_128_mouth/checkpoint/Audio2Face/model_finetune_lr1e-4_gamma_99_bs_128_200.pth', map_location=torch.device('cpu'))
state_dict = torch.load('/home/ubuntu/Audio2Face/output_mine_other/checkpoint/Audio2Face/modeltest200.pth', map_location=torch.device('cpu'))
# print([k for k, v in state_dict.items()])
model.load_state_dict(state_dict)
model.eval()
torch_output = model(test_arr)
 
input_names = ['input']
output_names = ['output']
torch.onnx.export(model, 
                  dummy_input, 
                  "mobilenet_v2.onnx", 
                  export_params=True,
                  verbose=False, 
                  input_names=input_names, 
                  output_names=output_names,
                  opset_version=11)

# model = onnx.load("mobilenet_v2.onnx")
# ort_session = ort.InferenceSession('mobilenet_v2.onnx')
# onnx_outputs = ort_session.run(None, {'input': test_arr})
# print('{}{}Export ONNX!'.format(np.array(onnx_outputs[0]).shape, np.array(onnx_outputs[1]).shape))

# import onnx
# from onnxsim import simplify

# # load your predefined ONNX model
# model = onnx.load("mobilenet_v2.onnx")

# # convert model
# model, check = simplify(model)

# assert check, "Simplified ONNX model could not be validated"


from onnx_tf.backend import prepare
import onnx

TF_PATH = "tf_model" # where the representation of tensorflow model will be stored
ONNX_PATH = "mobilenet_v2.onnx" # path to my existing ONNX model
onnx_model = onnx.load(ONNX_PATH)  # load onnx model
tf_rep = prepare(onnx_model)  # creating TensorflowRep object
tf_rep.export_graph(TF_PATH)

import tensorflow as tf

TF_PATH = "tf_model" 
TFLITE_PATH = "other.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tf_lite_model = converter.convert()
with open(TFLITE_PATH, 'wb') as f:
    f.write(tf_lite_model)

