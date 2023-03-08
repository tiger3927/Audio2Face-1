#!/usr/bin/env sh

TASK='facegood'

DATE=`date +"%Y-%m-%d-%H-%M-%S"`

LOG='./logs/'$TASK'-'$DATE

GPUs=1

CPUs=24

NODE=1

PARTITION='digitalcity'

id='lr_1e-4_gamma_0.9_bs_128'
lr=0.0005
gamma=0.9
bs=128
epoch=200

dataset='dataSet4_16'
output_path='./output4_16'
output_feature='mouth'
model_path='./output4_16/checkpoint/Audio2Face/modeltest200.pth'

mkdir $LOG
srun --partition=$PARTITION --cpus-per-task $CPUs --mpi=pmi2 --gres=gpu:$GPUs -n1 --ntasks-per-node=$NODE --job-name=$Task --kill-on-bad-exit=1 python /mnt/petrelfs/chenkeyu/FACEGOOD-Audio2Face/code/my_train/train.py \
--id=$id \
--lr=$lr \
--gamma=$gamma \
--bs=$bs \
--epoch=$epoch \
--dataset=$dataset \
--output_path=$output_path \
--output_feature=$output_feature \
# --finetune \
--model_path=$model_path \
2>&1|tee $LOG/train-$DATE.log &