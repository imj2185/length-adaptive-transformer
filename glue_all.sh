#!/usr/bin/bash

MODEL=$1
# EXP_TYPE=$2
# BENCHMARK=$3
# run_squad conda env prune in A100

function standard_finetune () {
    echo $1 $2 $3 "${4,,}"
    #tmux send-keys -t "$1" "git checkout master" Enter
    if  [ $3 = "mobilebert" ]
    then
        tmux send-keys -t "$1" "CUDA_VISIBLE_DEVICES=$2 python run_glue.py --model_name_or_path google/$3-uncased --task_name "${4,,}" --do_train --do_eval --data_dir glue/$4 --max_seq_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir glue_output/$4/$3/standard --overwrite_output_dir" Enter
    elif [ $3 = "bert" ]
    then
        tmux send-keys -t "$1" "CUDA_VISIBLE_DEVICES=$2 python run_glue.py --model_name_or_path $3-base-cased --task_name "${4,,}" --do_train --do_eval --data_dir glue/$4 --max_seq_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir glue_output/$4/$3/standard --overwrite_output_dir" Enter
    elif [ $3 = "distilbert" ]
    then
        tmux send-keys -t "$1" "CUDA_VISIBLE_DEVICES=$2 python run_glue.py --model_name_or_path $3-base-cased --task_name "${4,,}" --do_train --do_eval --data_dir glue/$4 --max_seq_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir glue_output/$4/$3/standard --overwrite_output_dir" Enter
    fi
}

cuda_id=0
for bench in "CoLA" "MNLI" "QNLI" "QQP" "RTE" "SST-2" "STS-B" "WNLI" "MRPC"
do
    mkdir glue_output/$bench
    mkdir glue_output/$bench/$MODEL
    
    session=$bench
    session+=_
    session+=$MODEL

    #echo $session
    #echo $bench

    tmux new-session -d -s "$session"
    tmux send-keys -t "$session" "setenv PATH /home/mdl/cvl5361/softwares/a100/bin:/home/mdl/cvl5361/softwares/a100/condabin:/usr/local/cuda-11.4/bin:/home/grads/cvl5361/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:." Enter
    tmux send-keys -t "$session" "bash" Enter
    tmux send-keys -t "$session" "export PATH=/home/mdl/cvl5361/softwares/a100/bin:/home/mdl/cvl5361/softwares/a100/condabin:/usr/local/cuda-11.4/bin:/home/grads/cvl5361/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:." Enter
    tmux send-keys -t "$session" "conda activate lat" Enter

    standard_finetune $session $((cuda_id % 4)) $MODEL $bench
    let "cuda_id+=1"
done




