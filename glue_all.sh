#!/usr/bin/bash

MODEL=$1
EXP=$2
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

function lat_finetune () {
    echo $1 $2 $3 "${4,,}"
    #tmux send-keys -t "$1" "git checkout master" Enter
    if  [ $3 = "mobilebert" ]
    then
        tmux send-keys -t "$1" "CUDA_VISIBLE_DEVICES=$2 python run_glue.py --model_name_or_path glue_output/$4/$3/standard/checkpoint-best --task_name "${4,,}" --do_train --do_eval --data_dir glue/$4 --max_seq_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 5.0 --output_dir glue_output/$4/$3/length_adaptive --length_adaptive --num_sandwich 2 --length_drop_ratio_bound 0.05 --layer_dropout_prob 0.15 --overwrite_output_dir" Enter
    elif [ $3 = "bert" ]
    then
        tmux send-keys -t "$1" "CUDA_VISIBLE_DEVICES=$2 python run_glue.py --model_name_or_path glue_output/$4/$3/standard/checkpoint-best --task_name "${4,,}" --do_train --do_eval --data_dir glue/$4 --max_seq_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 5.0 --output_dir glue_output/$4/$3/length_adaptive --length_adaptive --num_sandwich 2 --length_drop_ratio_bound 0.2 --layer_dropout_prob 0.2 --overwrite_output_dir" Enter
    elif [ $3 = "distilbert" ]
    then
        tmux send-keys -t "$1" "CUDA_VISIBLE_DEVICES=$2 python run_glue.py --model_name_or_path glue_output/$4/$3/standard/checkpoint-best --task_name "${4,,}" --do_train --do_eval --data_dir glue/$4 --max_seq_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 5.0 --output_dir glue_output/$4/$3/length_adaptive --length_adaptive --num_sandwich 2 --length_drop_ratio_bound 0.2 --layer_dropout_prob 0.2 --overwrite_output_dir" Enter
    fi
}

function that_finetune () {
    echo $1 $2 $3 "${4,,}"
    #tmux send-keys -t "$1" "git checkout master" Enter
    if  [ $3 = "mobilebert" ]
    then
        tmux send-keys -t "$1" "CUDA_VISIBLE_DEVICES=$2 python run_glue.py --model_name_or_path glue_output/$4/$3/standard/checkpoint-best --task_name "${4,,}" --do_train --do_eval --data_dir glue/$4 --max_seq_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 5.0 --output_dir glue_output/$4/$3/joint_adaptive --length_adaptive --num_sandwich 2 --length_drop_ratio_bound 0.05 --layer_dropout_prob 0.15 --overwrite_output_dir" Enter
    elif [ $3 = "bert" ]
    then
        tmux send-keys -t "$1" "CUDA_VISIBLE_DEVICES=$2 python run_glue.py --model_name_or_path glue_output/$4/$3/standard/checkpoint-best --task_name "${4,,}" --do_train --do_eval --data_dir glue/$4 --max_seq_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 5.0 --output_dir glue_output/$4/$3/joint_adaptive --length_adaptive --num_sandwich 2 --length_drop_ratio_bound 0.2 --layer_dropout_prob 0.2 --overwrite_output_dir" Enter
    elif [ $3 = "distilbert" ]
    then
        tmux send-keys -t "$1" "CUDA_VISIBLE_DEVICES=$2 python run_glue.py --model_name_or_path glue_output/$4/$3/standard/checkpoint-best --task_name "${4,,}" --do_train --do_eval --data_dir glue/$4 --max_seq_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 5.0 --output_dir glue_output/$4/$3/joint_adaptive --length_adaptive --num_sandwich 2 --length_drop_ratio_bound 0.2 --layer_dropout_prob 0.2 --overwrite_output_dir" Enter
    fi
}

function lat_evo_search() {
    echo $1 $2 $3 "${4,,}"
    #tmux send-keys -t "$1" "git checkout master" Enter
    if  [ $3 = "mobilebert" ]
    then
        tmux send-keys -t "$1" "CUDA_VISIBLE_DEVICES=$2 python run_glue.py --model_name_or_path glue_output/$4/$3/length_adaptive/checkpoint-best --task_name "${4,,}" --do_search --data_dir glue/$4 --max_seq_length 128 --per_device_eval_batch_size 16 --output_dir glue_output/$4/$3/evolutionary_search_lat --evo_iter 30 --mutation_size 30 --crossover_size 30 --overwrite_output_dir" Enter
    elif [ $3 = "bert" ]
    then
        tmux send-keys -t "$1" "CUDA_VISIBLE_DEVICES=$2 python run_glue.py --model_name_or_path glue_output/$4/$3/length_adaptive/checkpoint-best --task_name "${4,,}" --do_search --data_dir glue/$4 --max_seq_length 128 --per_device_eval_batch_size 16 --output_dir glue_output/$4/$3/evolutionary_search_lat --evo_iter 30 --mutation_size 30 --crossover_size 30 --overwrite_output_dir" Enter
    elif [ $3 = "distilbert" ]
    then
        tmux send-keys -t "$1" "CUDA_VISIBLE_DEVICES=$2 python run_glue.py --model_name_or_path glue_output/$4/$3/length_adaptive/checkpoint-best --task_name "${4,,}" --do_search --data_dir glue/$4 --max_seq_length 128 --per_device_eval_batch_size 16 --output_dir glue_output/$4/$3/evolutionary_search_lat --evo_iter 30 --mutation_size 30 --crossover_size 30 --overwrite_output_dir" Enter
    fi
}

function that_evo_search() {
    echo $1 $2 $3 "${4,,}"
    #tmux send-keys -t "$1" "git checkout master" Enter
    if  [ $3 = "mobilebert" ]
    then
        tmux send-keys -t "$1" "CUDA_VISIBLE_DEVICES=$2 python run_glue.py --model_name_or_path glue_output/$4/$3/joint_adaptive/checkpoint-best --task_name "${4,,}" --do_search --data_dir glue/$4 --max_seq_length 128 --per_device_eval_batch_size 16 --output_dir glue_output/$4/$3/evolutionary_search_joint --evo_iter 30 --mutation_size 30 --crossover_size 30 --overwrite_output_dir" Enter
    elif [ $3 = "bert" ]
    then
        tmux send-keys -t "$1" "CUDA_VISIBLE_DEVICES=$2 python run_glue.py --model_name_or_path glue_output/$4/$3/joint_adaptive/checkpoint-best --task_name "${4,,}" --do_search --data_dir glue/$4 --max_seq_length 128 --per_device_eval_batch_size 16 --output_dir glue_output/$4/$3/evolutionary_search_joint --evo_iter 30 --mutation_size 30 --crossover_size 30 --overwrite_output_dir" Enter
    elif [ $3 = "distilbert" ]
    then
        tmux send-keys -t "$1" "CUDA_VISIBLE_DEVICES=$2 python run_glue.py --model_name_or_path glue_output/$4/$3/joint_adaptive/checkpoint-best --task_name "${4,,}" --do_search --data_dir glue/$4 --max_seq_length 128 --per_device_eval_batch_size 16 --output_dir glue_output/$4/$3/evolutionary_search_joint --evo_iter 30 --mutation_size 30 --crossover_size 30 --overwrite_output_dir" Enter
    fi
}

cuda_id=0
for bench in "CoLA" "MNLI" "QNLI" "QQP" "RTE" "SST-2" "STS-B" "MRPC"
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

    if  [ $EXP = "standard_finetune" ]
    then
        standard_finetune $session $((cuda_id % 4)) $MODEL $bench
    elif [ $EXP = "lat_finetune" ]
    then
        lat_finetune $session $((cuda_id % 4)) $MODEL $bench
    elif [ $EXP = "that_finetune" ]
    then
        that_finetune $session $((cuda_id % 4)) $MODEL $bench
    elif [ $EXP = "lat_evo_search" ]
    then
        lat_evo_search $session $((cuda_id % 4)) $MODEL $bench
    elif [ $EXP = "that_evo_search" ]
    then
        that_evo_search $session $((cuda_id % 4)) $MODEL $bench
    fi
    let "cuda_id+=1"
done




