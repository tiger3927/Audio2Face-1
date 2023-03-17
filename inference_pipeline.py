from inference import inference
from postprocess.post_utils import concat_mouth_other
import json
import os
from preprocess.convert_wav2npy import wav2npy

if __name__ == '__main__':
    seed = 0
    for fn in os.listdir('./tts_audio'):
        input_name = fn
        # Can be changed according to your needs
        input_wav = './tts_audio/{}/{}_tts.wav'.format(input_name, input_name)
        output_mouth_json = './tts_audio/{}/{}_mouth.json'.format(input_name, input_name)
        output_other_json = './tts_audio/{}/{}_other.json'.format(input_name, input_name)
        output_head_json = './tts_audio/{}/{}_head.json'.format(input_name, input_name)
        output_path = './tts_audio/{}/{}_total.json'.format(input_name, input_name)
        fps = 30
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
        model_path = './output_mine_head_lr_1e-4_gamma_0.99_bs_128_head_head\checkpoint\Audio2Face\model_lr_1e-4_gamma_0.99_bs_128_head_400.pth'
        output_head = inference(seed, model_path, x_test, output_head_json, 3, 'head', fps, head_larger)
        # Output (frames_num, 7)

        output_json = concat_mouth_other(output_mouth, output_other, output_head, fps, mouth_larger, other_smaller, head_larger)
        # Output {[frames_num, 52]}

        with open(output_path, 'w') as f:
            json.dump(output_json, f)
        print('json saved to', output_path, '\n')