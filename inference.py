import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import opt.opts as opts
import json

from preprocess.convert_wav2npy import wav2npy
from model.model import Audio2Face

def test(model, output_size, test_loader):
    """ Do inference
    Args:
        model: model
        output_size: int, output size, 27 for mouth, 24 for other
        test_loader: torch.utils.data.DataLoader
    """
    model.eval()
    output_data = np.zeros((1, output_size))
    for test_data in test_loader:
        with torch.no_grad():
            predictions, emotion_input = model(test_data)
            predictions = predictions.detach().numpy()
            output_data = np.vstack((output_data, predictions))

    output_data = output_data[1:]
    print('output shape', output_data.shape)

    return output_data

def dim52todict(dim52, ouptut_dir, fps):
    """ Convert dim52 to dict
    Args:
        dim52: np.array, (frame_num, 52)
        model_path: str, load model from
        input_wav: str, input wav path
        output_npy: str, output npy path
        output_size: int, output size, 27 for mouth, 24 for other
        fps: int, if is inference data, set the fps
    """
    bs_names = ['browInnerUp','browDown_L','browDown_R','browOuterUp_L','browOuterUp_R','eyeLookUp_L','eyeLookUp_R','eyeLookDown_L','eyeLookDown_R','eyeLookIn_L','eyeLookIn_R','eyeLookOut_L','eyeLookOut_R','eyeBlink_L','eyeBlink_R','eyeSquint_L','eyeSquint_R','eyeWide_L','eyeWide_R','cheekPuff','cheekSquint_L','cheekSquint_R','noseSneer_L','noseSneer_R','jawOpen','jawForward','jawLeft','jawRight','mouthFunnel','mouthPucker','mouthLeft','mouthRight','mouthRollUpper','mouthRollLower','mouthShrugUpper','mouthShrugLower','mouthClose','mouthSmile_L','mouthSmile_R','mouthFrown_L','mouthFrown_R','mouthDimple_L','mouthDimple_R','mouthUpperUp_L','mouthUpperUp_R','mouthLowerDown_L','mouthLowerDown_R','mouthPress_L','mouthPress_R','mouthStretch_L','mouthStretch_R','tongueOut']
    
    output_frames = []
    for frame in dim52:
        bs_weight = {}
        for num, name in zip(frame, bs_names):
            bs_weight[name] = num
        output_frames.append(bs_weight)

    output = {}
    output['data'] = []
    for i, frame in enumerate(output_frames):
        output['data'].append({"facialExpression": frame, "time": i * float(1.0 / fps), "headAngles" : [0, 0, 0]})

    with open(ouptut_dir, 'w') as f:
        json.dump(output, f)

def inference(seed, model_path, input_wav, output_npy, output_size, fps):
    """ Do inference from wav to npy
    Args:
        seed: int, random seed
        model_path: str, load model from
        input_wav: str, input wav path
        output_npy: str, output npy path
        output_size: int, output size, 27 for mouth, 24 for other
        fps: int, if is inference data, set the fps
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load data
    x_test = wav2npy(input_wav, fps = fps)
    print('test data len:', x_test.shape[0])
    
    # Convert to tensor
    x_test = torch.as_tensor(x_test, dtype=torch.float32)[:-1]

    # Dataset to DataLoader
    test_loader = torch.utils.data.DataLoader(
        dataset = x_test,
        batch_size=32,
        shuffle=False,
        num_workers=2,
    )

    # Load model
    model = Audio2Face(output_size)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # print([k for k, v in state_dict.items()])
    model.load_state_dict(state_dict)

    print('test begin')
    output_data = test(model, output_size, test_loader)

    np.save(output_npy, output_data)
    print(output_data.shape, 'npy array saved to', output_npy)
    print('test finished!')
    return output_data

if __name__ == '__main__':
    opt = opts.parse_opt_test()
    seed = opt.seed
    model_path = opt.model_path
    input_wav = opt.input_wav
    output_npy = opt.output_npy
    output_size = opt.output_size
    fps = opt.fps
    output_data = inference(seed, model_path, input_wav, output_npy, output_size, fps)
