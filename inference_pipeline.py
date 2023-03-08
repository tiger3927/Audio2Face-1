from inference import inference
from postprocess.post_utils import concat_mouth_other
import json
import os

if __name__ == '__main__':
    seed = 0
    for fn in os.listdir('./tts_audio'):
        input_name = fn
        # Can be changed according to your needs
        input_wav = './tts_audio/{}/{}_tts.wav'.format(input_name, input_name)
        output_mouth_json = './tts_audio/{}/{}_mouth.json'.format(input_name, input_name)
        output_other_json = './tts_audio/{}/{}_other.json'.format(input_name, input_name)
        fps = 30
        
        # Test mouth
        model_path = './output_4_16_mine_finetune_lr1e-4_gamma_99_bs_128_mouth/checkpoint/Audio2Face/model_finetune_lr1e-4_gamma_99_bs_128_200.pth'
        output_mouth = inference(seed, model_path, input_wav, output_mouth_json, 27, 'mouth', fps)
        # Output (frames_num, 27)

        # Test other
        model_path = './output_mine_other/checkpoint/Audio2Face/modeltest200.pth'
        output_other = inference(seed, model_path, input_wav, output_other_json, 24, 'other', fps)
        # Output (frames_num, 24)

        output_json = concat_mouth_other(output_mouth, output_other, fps)
        # Output {[frames_num, 52]}

        output_path = './tts_audio/{}/{}.json'.format(input_name, input_name)
        with open(output_path, 'w') as f:
            json.dump(output_json, f)
        print('json saved to', output_path, '\n')