gpu=6

id='lr_1e-3_gamma_0.9_bs_32_head'
log="$id.log"
lr=0.001
gamma=0.9
bs=32
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