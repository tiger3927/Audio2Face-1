from postprocess.post_utils import concat_mouth_other, save_separately
import json
import os
from preprocess.convert_wav2npy import wav2npy
import tensorflow as tf
import numpy as np
import time

import opt.opts as opts

if __name__ == '__main__':
    opt = opts.parse_opt_test()
    seed = opt.seed
    model_path = opt.model_path
    input_wav = opt.input_wav

    # Can be changed according to your needs
    input_wav = input_wav
    output_mouth_json = input_wav.replace('.wav','_mouth.json')
    output_other_json = input_wav.replace('.wav','_other.json')
    output_head_json = input_wav.replace('.wav','_head.json')
    output_path = input_wav.replace('.wav','_total.json')
    fps = opt.fps
    # scale
    mouth_larger = 1.2
    other_smaller = 0.5
    head_larger = 3.14
    
    # preprocess data
    x_test = wav2npy(input_wav, fps = fps)
    
    # save preprocess data
    np.save(input_wav.replace('.wav','_preprocess.npy'), x_test)

    # Test mouth
    mouth_model_path = './mouth.tflite'
    other_model_path = './other.tflite'
    head_model_path = './head.tflite'
    # load tf model
    interpreter_m = tf.lite.Interpreter(mouth_model_path)
    interpreter_o = tf.lite.Interpreter(other_model_path)
    interpreter_h = tf.lite.Interpreter(head_model_path)
    interpreter_m.allocate_tensors()
    interpreter_o.allocate_tensors()
    interpreter_h.allocate_tensors()

    # get input and output tensor
    input_details = interpreter_m.get_input_details()
    output_details = interpreter_m.get_output_details()
    print(input_details)
    print(output_details)

    # test frames
    # x_test = np.zeros((1130, 32, 64, 1), dtype=np.float32)

    # add padding and split to batch
    frame_num = x_test.shape[0]
    pad = 100 - (x_test.shape[0] % 100)
    print("frame_num:", frame_num, "padding:", pad)
    x_test = np.append(x_test, np.zeros((pad, 32, 64, 1), dtype=np.float32), axis=0)
    x_test_batchs = np.split(x_test, x_test.shape[0]/100, axis=0)

    output_mouth = np.zeros((1, 27))
    output_other = np.zeros((1, 24))
    output_head = np.zeros((1, 3))
    
    for test_arr in x_test_batchs:
        b_time = time.time()
        # set input
        interpreter_m.set_tensor(input_details[0]['index'], test_arr.astype(np.float32))
        interpreter_o.set_tensor(input_details[0]['index'], test_arr.astype(np.float32))
        interpreter_h.set_tensor(input_details[0]['index'], test_arr.astype(np.float32))

        # inference
        interpreter_m.invoke()
        interpreter_o.invoke()
        interpreter_h.invoke()

        # get output
        print(interpreter_m.get_tensor(output_details[1]['index']).shape)
        output_mouth = np.append(output_mouth, interpreter_m.get_tensor(output_details[1]['index']), axis=0)
        output_other = np.append(output_other, interpreter_o.get_tensor(output_details[1]['index']), axis=0)
        output_head = np.append(output_head, interpreter_h.get_tensor(output_details[1]['index']), axis=0)
        print(time.time()-b_time)

    # cut output
    output_mouth = output_mouth[1:frame_num+1]
    output_other = output_other[1:frame_num+1]
    output_head = output_head[1:frame_num+1]

    json_data = save_separately(output_mouth, 'mouth', fps, mouth_larger)
    json.dump(json_data, open(output_mouth_json, 'w'))

    json_data = save_separately(output_other, 'other', fps, other_smaller)
    json.dump(json_data, open(output_other_json, 'w'))

    json_data = save_separately(output_head, 'head', fps, head_larger)
    json.dump(json_data, open(output_head_json, 'w'))

    # concat output
    output_json = concat_mouth_other(output_mouth, output_other, output_head, fps, mouth_larger, other_smaller, head_larger)
    # Output (frames_num, 52)

    with open(output_path, 'w') as f:
        json.dump(output_json, f)
    print('json saved to', output_path, '\n')