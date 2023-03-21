gpu=7

id='lr_1e-4_gamma_0.99_bs_128_head'
log="$id.log"
lr=0.0001
gamma=0.99
bs=128
epoch=400

dataset='dataSet_mine'
output_path='./output_mine_head'
output_feature='head'
# model_path='./output4_16/checkpoint/Audio2Face/modeltest200.pth'

nohup python train.py \
--gpu=$gpu \
--id=$id \
--lr=$lr \
--gamma=$gamma \
--bs=$bs \
--epoch=$epoch \
--dataset=$dataset \
--output_path=$output_path \
--output_feature=$output_feature > $log 2>&1 &
# --finetune \
# --model_path=$model_path