gpu=7

id='finetune_8_lr1e-4_gamma_99_bs_128'
log="$id.log"
lr=0.0001
gamma=0.99
bs=128
epoch=400

dataset='dataSet_mine'
output_path='./output_4_16_mine'
output_feature='mouth'
model_path='./output4_16_lr_1e-4_gamma_0.99_bs_8_mouth/checkpoint/Audio2Face/model_lr_1e-4_gamma_0.99_bs_8_180.pth'

mkdir $LOG
python train.py \
--gpu=$gpu \
--id=$id \
--lr=$lr \
--gamma=$gamma \
--bs=$bs \
--epoch=$epoch \
--dataset=$dataset \
--output_path=$output_path \
--output_feature=$output_feature \
--finetune \
--model_path=$model_path > $log 2>&1 &