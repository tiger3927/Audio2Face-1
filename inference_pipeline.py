from inference import inference
from postprocess.post_utils import concat_mouth_other
import json
import os
from preprocess.convert_wav2npy import wav2npy

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
    mouth_larger = 1.2
    other_smaller = 0.5
    head_larger = 3.14
    

    x_test = wav2npy(input_wav, fps = fps)
    # Test mouth
    model_path = './output_4_16_mine_finetune_lr1e-4_gamma_99_bs_128_mouth/checkpoint/Audio2Face/model_finetune_lr1e-4_gamma_99_bs_128_200.pth'
    output_mouth = inference(seed, model_path, x_test, output_mouth_json, 27, 'mouth', fps, mouth_larger)
    # Output (frames_num, 27)

    # Test other
    model_path = './output_mine_other/checkpoint/Audio2Face/modeltest200.pth'
    output_other = inference(seed, model_path, x_test, output_other_json, 24, 'other', fps, other_smaller)
    # Output (frames_num, 24)

    # Test other
    model_path = './output_mine_head_lr_1e-4_gamma_0.99_bs_128_head_head/checkpoint/Audio2Face/model_lr_1e-4_gamma_0.99_bs_128_head_400.pth'
    output_head = inference(seed, model_path, x_test, output_head_json, 3, 'head', fps, head_larger)
    # Output (frames_num, 7)

    output_json = concat_mouth_other(output_mouth, output_other, output_head, fps, mouth_larger, other_smaller, head_larger)
    # Output {[frames_num, 52]}

    with open(output_path, 'w') as f:
        json.dump(output_json, f)
    print('json saved to', output_path, '\n')