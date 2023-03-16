gpu=6

id='lr_1e-4_gamma_0.99_bs_32_new'
log="$id.log"
lr=0.0001
gamma=0.99
bs=32
epoch=200

dataset='dataSet4_16'
output_path='./output4_16'
output_feature='mouth'
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