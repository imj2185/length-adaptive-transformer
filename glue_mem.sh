#!/bin/bash
# ------------------------------------------
SEED=$1
INPUT=memory_footprint.csv
OLDIFS=$IFS
IFS=','
cuda_id=$SEED
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read model bench finetune gene
do
    if [ $bench != "SQuADv1.1" ]
    then
        echo "Session : $model+$bench+$finetune"
        #echo $gene | tr \(\) _

        tmux new-session -d -s "$model+$bench+$finetune"
        tmux send-keys -t "$model+$bench+$finetune" "setenv PATH /home/mdl/cvl5361/softwares/a100/bin:/home/mdl/cvl5361/softwares/a100/condabin:/usr/local/cuda-11.4/bin:/home/grads/cvl5361/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:." Enter
        tmux send-keys -t "$model+$bench+$finetune" "bash" Enter
        tmux send-keys -t "$model+$bench+$finetune" "export PATH=/home/mdl/cvl5361/softwares/a100/bin:/home/mdl/cvl5361/softwares/a100/condabin:/usr/local/cuda-11.4/bin:/home/grads/cvl5361/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:." Enter
        tmux send-keys -t "$model+$bench+$finetune" "conda activate lat" Enter
        tmux send-keys "$model+$bench+$finetune" "CUDA_VISIBLE_DEVICES=$((cuda_id% 4)) python run_glue.py --model_name_or_path glue_output/$bench/${model,,}/standard/checkpoint-best --task_name "${bench,,}" --do_mem_track --do_eval --data_dir glue/$bench --max_seq_length 128 --per_device_eval_batch_size 16 --output_dir glue_output/$bench/${model,,}/mem --evo_iter 30 --mutation_size 30 --crossover_size 30 --test_gene $gene"
        #tmux send-keys -t "$model+$bench+$finetune" "CUDA_VISIBLE_DEVICES=$cuda_id python run_glue.py --model_name_or_path glue_output/$bench/${model,,}/standard/checkpoint-best --task_name "${bench,,}" --do_mem_track --do_eval --data_dir glue/$bench --max_seq_length 128 --per_device_eval_batch_size 16 --output_dir glue_output/$bench/${model,,}/mem --evo_iter 30 --mutation_size 30 --crossover_size 30 --test_gene $gene" Enter
        let "cuda_id+=1"

    fi
done < $INPUT
IFS=$OLDIFS