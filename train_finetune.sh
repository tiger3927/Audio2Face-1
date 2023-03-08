#!/usr/bin/env sh

TASK='facegood'

DATE=`date +"%Y-%m-%d-%H-%M-%S"`

LOG='./logs/'$TASK'-'$DATE

GPUs=1

CPUs=24

NODE=1

PARTITION='digitalcity'

id='finetune_128_200_lr1e-4_gamma_99_bs_128'
lr=0.0001
gamma=0.99
bs=128
epoch=400

dataset='dataSet_mine'
output_path='./output_4_16_mine'
output_feature='mouth'
model_path='./output4_16_lr_1e-4_gamma_0.9_bs_128_mouth/checkpoint/Audio2Face/model_lr_1e-4_gamma_0.9_bs_128_200.pth'

mkdir $LOG
srun --partition=$PARTITION --cpus-per-task $CPUs --mpi=pmi2 --gres=gpu:$GPUs -n1 --ntasks-per-node=$NODE -J=$Task --kill-on-bad-exit=1 python /mnt/petrelfs/chenkeyu/FACEGOOD-Audio2Face/code/my_train/train.py \
--id=$id \
--lr=$lr \
--gamma=$gamma \
--bs=$bs \
--epoch=$epoch \
--dataset=$dataset \
--output_path=$output_path \
--output_feature=$output_feature \
--finetune \
--model_path=$model_path \
2>&1|tee $LOG/train-$DATE.log &