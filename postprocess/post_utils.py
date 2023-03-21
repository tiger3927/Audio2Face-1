import json
import os
import numpy as np
import random

def frames_avg(input, type):
    """ Use conv to make frames smoother
    Args:
        input: np.array, in shape (frame_num, bs_weight_num)
        type: str, 'mouth' or 'other'
    Return:
        output: np.array, in shape (frame_num, bs_weight_num)
    """
    # Kernels for conv
    kernel_mouth = np.array([0.2, 0.6, 0.2])
    kernel_other = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1]) / 3
    kernel_head = np.ones((15,), np.float32) / 15

    output = np.zeros((input.shape[0], 1))
    # Split input into different dims
    for i in range(input.shape[1]):
        if type == 'mouth':
            output_f = np.convolve(input[:, i], kernel_mouth, mode="same")
        elif type == 'other':
            output_f = np.convolve(input[:, i], kernel_other, mode="same")
        elif type == 'head':
            output_f = np.convolve(input[:, i], kernel_head, mode="same")
        output_f = np.expand_dims(output_f, axis=1)
        output = np.hstack((output, output_f))

    return output[:, 1:]

def random_blink(dim52, bs_names, fps):
    """ Random blink
    Args:
        dim52: np.array, (frame_num, 52)
        bs_names: list, blend shape weight
    Return:
        output: dict, can be saved as json
    """
    blink_avg_time = 5
    num_count = 0
    output_frames = []
    for frame in dim52:
        # Set weight to proper name in dict
        bs_weight = {}
        for num, name in zip(frame, bs_names):
            bs_weight[name] = num

        # Random cost and blink
        blink_avg_time -= random.uniform(0.6, 1.4) / fps
        if blink_avg_time < 0:
            bs_weight['eyeBlink_L'] = 0.8
            bs_weight['eyeBlink_R'] = 0.8
            num_count += 1
            if num_count >= 3:
                blink_avg_time = 5
                num_count = 0
        output_frames.append(bs_weight)
    return output_frames

def save_separately(data, location, fps, scale = 1):
    """ Save mouth(27 dims) or other (24 dims) separately
    Args:
        data: np.array, (frame_num, 27)
        location: str, mouth
        fps: int, 30 or 60
    Return:
        output: dict, can be saved as json
    """
    bs_names = ['browInnerUp','browDown_L','browDown_R','browOuterUp_L','browOuterUp_R','eyeLookUp_L','eyeLookUp_R','eyeLookDown_L','eyeLookDown_R','eyeLookIn_L','eyeLookIn_R','eyeLookOut_L','eyeLookOut_R','eyeBlink_L','eyeBlink_R','eyeSquint_L','eyeSquint_R','eyeWide_L','eyeWide_R','cheekPuff','cheekSquint_L','cheekSquint_R','noseSneer_L','noseSneer_R','jawOpen','jawForward','jawLeft','jawRight','mouthFunnel','mouthPucker','mouthLeft','mouthRight','mouthRollUpper','mouthRollLower','mouthShrugUpper','mouthShrugLower','mouthClose','mouthSmile_L','mouthSmile_R','mouthFrown_L','mouthFrown_R','mouthDimple_L','mouthDimple_R','mouthUpperUp_L','mouthUpperUp_R','mouthLowerDown_L','mouthLowerDown_R','mouthPress_L','mouthPress_R','mouthStretch_L','mouthStretch_R','tongueOut']

    print(location, 'data shape', data.shape)

    # Smooth
    if location == 'mouth':
        # data = data * 1.5
        data = frames_avg(data, 'mouth') * scale

        # add zeros
        other_data = np.zeros((data.shape[0], 24))
        tough_data = np.zeros((data.shape[0], 1))
        dim52 = np.hstack((other_data, data, tough_data))

        output_frames = []
        for frame in dim52:
            # Set weight to proper name in dict
            bs_weight = {}
            for num, name in zip(frame, bs_names):
                bs_weight[name] = num
            output_frames.append(bs_weight)
    elif location == 'other':
        data = frames_avg(data, 'other') * scale

        # add zeros
        mouth_tough_data = np.zeros((data.shape[0], 28))
        dim52 = np.hstack((data, mouth_tough_data))
        
        # Random blink average time
        output_frames = random_blink(dim52, bs_names, fps)
    elif location == 'head':
        data = frames_avg(data, 'head') * scale 

        output_frames = data.tolist()

        output = {}
        output['data'] = []
        for i, frame in enumerate(output_frames):
            output['data'].append({"facialExpression": {bs_name: 0.0 for bs_name in bs_names}, "time": i * float(1.0 / fps), "headAngles" : frame})
        return output
        
    print(location, 'data shape after concat', dim52.shape)
        
    # Align by frames and set fps
    output = {}
    output['data'] = []
    for i, frame in enumerate(output_frames):
        output['data'].append({"facialExpression": frame, "time": i * float(1.0 / fps), "headAngles" : [0, 0, 0]})

    return output

def concat_mouth_other(mouth_data, other_data, head_data, fps, mouth_larger = 1, other_smaller = 1, head_larger = 1):
    """Concat mouth(27 dims) and other (24 dims)
    Args:
        mouth_data: np.array, (frame_num, 27)
        other_data: np.array, (frame_num, 24)
        other_data: np.array, (frame_num, 3)
        fps: int, 30 or 60
    Return:
        output: dict, can be saved as json
    """
    bs_names = ['browInnerUp','browDown_L','browDown_R','browOuterUp_L','browOuterUp_R','eyeLookUp_L','eyeLookUp_R','eyeLookDown_L','eyeLookDown_R','eyeLookIn_L','eyeLookIn_R','eyeLookOut_L','eyeLookOut_R','eyeBlink_L','eyeBlink_R','eyeSquint_L','eyeSquint_R','eyeWide_L','eyeWide_R','cheekPuff','cheekSquint_L','cheekSquint_R','noseSneer_L','noseSneer_R','jawOpen','jawForward','jawLeft','jawRight','mouthFunnel','mouthPucker','mouthLeft','mouthRight','mouthRollUpper','mouthRollLower','mouthShrugUpper','mouthShrugLower','mouthClose','mouthSmile_L','mouthSmile_R','mouthFrown_L','mouthFrown_R','mouthDimple_L','mouthDimple_R','mouthUpperUp_L','mouthUpperUp_R','mouthLowerDown_L','mouthLowerDown_R','mouthPress_L','mouthPress_R','mouthStretch_L','mouthStretch_R','tongueOut']
    
    print('mouth data shape', mouth_data.shape)
    print('other data shape', other_data.shape)
    print('other data shape', head_data.shape)

    # Smooth
    mouth_data = frames_avg(mouth_data, 'mouth') * mouth_larger
    other_data = frames_avg(other_data * other_smaller, 'other')
    head_data = frames_avg(head_data * head_larger, 'head') 
    head_data = head_data.tolist()

    # Concat mouth and other weight (frames_num, 52)
    dim52 = np.hstack((other_data, mouth_data))
    tough_data = np.zeros((other_data.shape[0], 1))
    dim52 = np.hstack((dim52, tough_data))

    # Random blink average time
    output_frames = random_blink(dim52, bs_names, fps)

    # Align by frames and set fps
    output = {}
    output['data'] = []
    for i, (frame, h) in enumerate(zip(output_frames, head_data)):
        output['data'].append({"facialExpression": frame, "time": i * float(1.0 / fps), "headAngles" : h})

    return output

if __name__ == '__main__':
    mouth_data_path = ''
    other_data_path = ''
    output_path = ''
    fps = 30

    print('read mouth data form', mouth_data_path)
    mouth_data = np.load(mouth_data_path)
    print('read other data form', other_data_path)
    other_data = np.load(other_data_path)
    
    output_json = concat_mouth_other(mouth_data, other_data, fps)
    
    with open(output_path, 'r') as f:
        json.dump(output_json, f)