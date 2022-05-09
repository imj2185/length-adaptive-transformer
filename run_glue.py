# coding=utf-8
# Length-Adaptive Transformer
# Copyright (c) 2020-present NAVER Corp.
# Apache License v2.0
#####
# Original code is from https://github.com/huggingface/transformers
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""


import dataclasses
import logging
import os

from regex import I
#os.environ["CUDA_VISIBLE_DEVICES"]='1'
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

from length_adaptive_transformer import (
    TrainingArguments,
    LengthDropArguments,
    SearchArguments,
    LengthDropTrainer,
)
from length_adaptive_transformer.drop_and_restore_utils import (
    sample_length_configuration,
    sample_head_configuration,
)
from length_adaptive_transformer.evolution import (
    approx_ratio, inverse, store2str
)

logger = logging.getLogger(__name__)

glue_tasks_metrics = {
    "cola": "mcc",
    "mnli": "mnli/acc",
    "mnli-mm": "mnli-mm/acc",
    "mrpc": "acc_and_f1",
    "sst-2": "acc",
    "sts-b": "pearson_and_spearman",
    "qqp": "acc_and_f1",
    "qnli": "acc",
    "rte": "acc",
    "wnli": "acc",
}

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, LengthDropArguments, SearchArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, length_drop_args, search_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, length_drop_args, search_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        # format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        format="%(asctime)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval or search_args.do_search
        else None
    )
    test_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
        if training_args.do_predict
        else None
    )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            if output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            else:  # regression
                preds = np.squeeze(preds)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    if search_args.do_search:
        model.config.output_attentions = True

    assert not (length_drop_args.length_config and length_drop_args.length_adaptive)
    if length_drop_args.length_adaptive or search_args.do_search:
        training_args.max_seq_length = data_args.max_seq_length

    # Initialize our Trainer
    model.set_fn_layer_parameters()
    trainer = LengthDropTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        best_metric=glue_tasks_metrics[data_args.task_name],
        length_drop_args=length_drop_args,
    )

    # Training
    if training_args.do_train:
        global_step, best = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        best_msg = ", ".join([f"{k} {v}" for k, v in best.items()])
        logger.info(f" global_step = {global_step} | best: {best_msg}")
        '''
        output_dir = os.path.join(training_args.output_dir, "checkpoint-last")
        trainer.save_model(output_dir)
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(output_dir)
        '''

    # Evaluation
    eval_results = {}
    if training_args.do_eval and not search_args.do_search:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
            )

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result, _ = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    if training_args.do_predict:
        logger.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            )

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))

    # Search
    if search_args.do_search:
        import warnings
        warnings.filterwarnings("ignore")

        output_dir = training_args.output_dir

        # assert args.population_size == args.parent_size + args.mutation_size + args.crossover_size
        trainer.init_evolution()
        trainer.load_store(os.path.join(model_args.model_name_or_path, 'store.tsv'))

        lower_gene = sample_length_configuration(
            data_args.max_seq_length,
            config.num_hidden_layers,
            length_drop_ratio=length_drop_args.length_drop_ratio_bound,
        ), sample_head_configuration(
            config.num_attention_heads,
            config.num_hidden_layers,
            max_head_pruning=True,
        )

        upper_gene = (data_args.max_seq_length,) * config.num_hidden_layers, (0,) * config.num_hidden_layers
        trainer.add_gene(lower_gene, method=0, latency=search_args.latency_constraint)
        trainer.add_gene(upper_gene, method=0, latency=search_args.latency_constraint)
        trainer.lower_constraint = trainer.store[lower_gene][0]
        trainer.upper_constraint = trainer.store[upper_gene][0]

        length_drop_ratios = [inverse(r) for r in np.linspace(approx_ratio(length_drop_args.length_drop_ratio_bound), 1, search_args.population_size + 2)[1:-1]]
        head_drop_ratios = [r for r in np.linspace(1, config.num_attention_heads, search_args.population_size + 2)[1:-1]]
        
        for i in range(len(length_drop_ratios)):
            gene = sample_length_configuration(
                data_args.max_seq_length,
                config.num_hidden_layers,
                length_drop_ratio=length_drop_ratios[i],
            ), sample_head_configuration(
                config.num_attention_heads,
                config.num_hidden_layers,
                max_head_pruning=True,
                prune_ratio=head_drop_ratios[i],
            )
            trainer.add_gene(gene, method=0, latency=search_args.latency_constraint)

        for i in range(search_args.evo_iter + 1):
            logger.info(f"| Start Iteration {i}:")
            population, area = trainer.pareto_frontier()
            parents = trainer.convex_hull()
            results = {"area": area, "population_size": len(population), "num_parents": len(parents)}

            logger.info(f"| >>>>>>>> {' | '.join([f'{k} {v}' for k, v in results.items()])}")
            for gene in parents:  # population
                logger.info("| " + store2str(gene, *trainer.store[gene][:3]))

            trainer.save_store(os.path.join(output_dir, f'store-iter{i}.tsv'))
            trainer.save_population(os.path.join(output_dir, f'population-iter{i}.tsv'), population)
            trainer.save_population(os.path.join(output_dir, f'parents-iter{i}.tsv'), parents)

            if i == search_args.evo_iter:
                break

            k = 0
            while k < search_args.mutation_size:
                if trainer.mutate(search_args.mutation_prob, search_args.latency_constraint):
                    k += 1

            k = 0
            while k < search_args.crossover_size:
                if trainer.crossover(search_args.latency_constraint):
                    k += 1

    return eval_results



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
