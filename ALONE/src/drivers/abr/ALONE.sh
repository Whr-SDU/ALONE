#!/bin/bash

set -e

save_dir=/home/ubuntu/Whr/EAS/ALONE/results_ALONE
video_size_file_dir=/home/ubuntu/Whr/EAS/ALONE/data/abr/video_sizes
val_trace_dir=/home/ubuntu/Whr/Load_trace/shiyan_2_val
total_epoch=25000
train_trace_dir=/home/ubuntu/Whr/Load_trace/train
pretrained_model=/home/ubuntu/Whr/EAS/ALONE/results_ALONE/4/gener/model_saved/nn_model_ep_24500.ckpt
master_trace_path=/home/ubuntu/Whr/EAS/new_val/results_4G/1/master_trace
#        --model-path ${pretrained_model} \
#        --entropy-weight 0.08322155707037684 \

seed=30
#for seed in 10 20 30; do
    python /home/ubuntu/Whr/EAS/ALONE/src/simulator/master_QoE_4G/ALONE/train_master_gener.py  \
        --save-dir ${save_dir}/w_0.5/gener \
        --seed ${seed} \
        --total-epoch ${total_epoch} \
        --model-path ${pretrained_model} \
        --entropy-weight 0.14684109696064138 \
        --video-size-file-dir ${video_size_file_dir} \
        --master-trace-path ${master_trace_path} \
        --nagent 20 \
        --train-trace-dir ${train_trace_dir} \
        --val-trace-dir ${val_trace_dir}
#done
