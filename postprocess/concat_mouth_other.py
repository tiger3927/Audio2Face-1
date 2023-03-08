import json
import os
import numpy as np
import random

def concat_mouth_other(mouth_data, other_data, fps):
    """Concat mouth(27 dims) and other (24 dims)
    Args:
        mouth_data_path: np.array, (frame_num, 27)
        other_data_path: np.array, (frame_num, 24)
        fps: int, 30 or 60
    Return:
        output: dict, can be saved as json
    """
    bs_names = ['browInnerUp','browDown_L','browDown_R','browOuterUp_L','browOuterUp_R','eyeLookUp_L','eyeLookUp_R','eyeLookDown_L','eyeLookDown_R','eyeLookIn_L','eyeLookIn_R','eyeLookOut_L','eyeLookOut_R','eyeBlink_L','eyeBlink_R','eyeSquint_L','eyeSquint_R','eyeWide_L','eyeWide_R','cheekPuff','cheekSquint_L','cheekSquint_R','noseSneer_L','noseSneer_R','jawOpen','jawForward','jawLeft','jawRight','mouthFunnel','mouthPucker','mouthLeft','mouthRight','mouthRollUpper','mouthRollLower','mouthShrugUpper','mouthShrugLower','mouthClose','mouthSmile_L','mouthSmile_R','mouthFrown_L','mouthFrown_R','mouthDimple_L','mouthDimple_R','mouthUpperUp_L','mouthUpperUp_R','mouthLowerDown_L','mouthLowerDown_R','mouthPress_L','mouthPress_R','mouthStretch_L','mouthStretch_R','tongueOut']
    
    print('mouth data shape', mouth_data.shape)
    print('other data shape', other_data.shape)

    # Smooth
    other_data = frames_avg(other_data * 0.5, 'other')
    mouth_data = frames_avg(mouth_data, 'mouth')

    # Concat mouth and other weight (frames_num, 52)
    dim52 = np.hstack((other_data, mouth_data))
    tough_data = np.zeros((other_data.shape[0], 1))
    dim52 = np.hstack((dim52, tough_data))

    # Random blink average time
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

    # Align by frames and set fps
    output = {}
    output['data'] = []
    for i, frame in enumerate(output_frames):
        output['data'].append({"facialExpression": frame, "time": i * float(1.0 / fps), "headAngles" : [0, 0, 0]})

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