# Audio to Face

## Environment Setup
You can install python library as you need or use the requirement.txt to help you with the environment. We recommend python=3.9.
```
pip install numpy torch scipy librosa 
```
or
```
pip install -r requirements.txt
```

## Train
The train shell code needs to be run on the sever. We will not specify it here.
```
python train.py
```

### Data Preprocessing
As we are provided with blendshape weight saved in txt and audio saved in wav, we can use the command as follows to read the data and conver it to numpy format. Then, we can use propare_data.py to concat them together and split them to training and testing sets. (Remember to change the file path)
```
python preprocess/convert_txt2npy.py
python preprocess/convert_wav2npy.py
python preprocess/prepare_data.py
```

## Test
### Inference data together (Remember to change the file path)
```
python inference_pipeline.py
```
### Inference mouth data
```
python inference.py \
--seed 0 \
--location mouth \
--input_wav ./tts_audio/20230224154812/20230224154812_tts.wav \
--fps 30 \
--model_path ./output_4_16_mine_finetune_lr1e-4_gamma_99_bs_128_mouth/checkpoint/Audio2Face/model_finetune_lr1e-4_gamma_99_bs_128_200.pth \
--output_json ./tts_audio/20230224154812/20230224154812_mouth.json \
--output_size 27 \
--scale 1.2
```
### Inference other data
```
python inference.py \
--seed 0 \
--location other \
--input_wav ./tts_audio/20230224154812/20230224154812_tts.wav \
--fps 30 \
--model_path ./output_mine_other/checkpoint/Audio2Face/modeltest200.pth \
--output_json ./tts_audio/20230224154812/20230224154812_other.json \
--output_size 24 \
--scale 0.5
```
### Inference head data
```
python inference.py \
--seed 0 \
--location head \
--input_wav ./tts_audio/20230224154812/20230224154812_tts.wav \
--fps 30 \
--model_path ./C:\Users\yangmingzhuo\FACEGOOD-Audio2Face\code\my_train\output_mine_head_lr_1e-4_gamma_099_bs_128_head_head\checkpoint\Audio2Face\model_lr_1e-4_gamma_0.99_bs_128_head_400.pth \
--output_json ./tts_audio/20230224154812/20230224154812_head.json \
--output_size 3 \
--scale 3.14
```
### Concat data (Remember to change the file path)
```
python postprocess/concat_mouth_other.py
```
## Experiment Log
[Finetune Tencent Document](https://docs.qq.com/sheet/DYVlDcXhEb2RBSHN6?tab=BB08J2&u=02639db8698c4a47991e544165bdf1c0)