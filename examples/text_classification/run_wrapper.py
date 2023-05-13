"""Wrapper launcher script."""

import os

import fire

from .src import common


def _get_command(
    task_name,
    output_dir,
    model_name_or_path,
    data_dir,
    learning_rate,
    clipping_mode: str,
    clipping_fn: str,
    clipping_style: str,
    non_private,
    target_epsilon,
    few_shot_type,
    seed,
    attention_only,
    bias_only,
    static_lm_head,
    static_embedding,
    randomly_initialize,
    physical_batch_size,
    batch_size,
    num_train_epochs,
    eval_steps,
):
    task_name_to_factor = {
        "sst-2": 1, "qnli": 2, "qqp": 6, "mnli": 6,
    }
    factor = task_name_to_factor[task_name]

    if batch_size is None:
        base_batch_size = 1000
        # This batch size selection roughly ensures the sampling rates on different
        # datasets are in the same ballpark.
        batch_size = int(base_batch_size * factor)
    gradient_accumulation_steps = batch_size // physical_batch_size

    if num_train_epochs is None:
        base_num_train_epochs = 3
        num_train_epochs = int(base_num_train_epochs * factor)

    if learning_rate is None:
        if non_private.lower() in ('yes', 'y', 'true', 't'):
            learning_rate = 5e-5
            if bias_only.lower() in ('yes', 'y', 'true', 't'):
                learning_rate=1e-3
        else:
            learning_rate = 5e-4
            if bias_only.lower() in ('yes', 'y', 'true', 't'):
                learning_rate=5e-3

    data_dir = f"{data_dir}/{common.task_name2suffix_name[task_name]}"
    template = {
        "sst-2": "*cls**sent_0*_It_was*mask*.*sep+*",
        "mnli": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
        "qnli": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
        "qqp": "*cls**sent-_0**mask*,*+sentl_1**sep+*",
    }[task_name]

    # Epochs chosen roughly to match e2e number of updates. We didn't hyperparameter tune on classification tasks :)
    cmd = f'''
python -m text_classification.run_classification \
  --task_name {task_name} \
  --data_dir {data_dir} \
  --output_dir {output_dir} \
  --overwrite_output_dir \
  --model_name_or_path {model_name_or_path} \
  --few_shot_type {few_shot_type} \
  --num_k 1 \
  --num_sample 1 --seed {seed} \
  --template {template} \
  --non_private {non_private} \
  --num_train_epochs {num_train_epochs} \
  --target_epsilon {target_epsilon} \
  --per_device_train_batch_size {physical_batch_size} \
  --gradient_accumulation_steps {gradient_accumulation_steps} \
  --per_device_eval_batch_size 8 \
  --per_example_max_grad_norm 0.1 --clipping_mode {clipping_mode} \
  --clipping_fn {clipping_fn} --clipping_style {clipping_style}\
  --learning_rate {learning_rate} \
  --lr_decay yes \
  --adam_epsilon 1e-08 \
  --weight_decay 0 \
  --max_seq_len 256 \
  --evaluation_strategy steps --eval_steps {eval_steps} --evaluate_before_training True \
  --do_train --do_eval \
  --first_sent_limit 200 --other_sent_limit 200 --truncate_head yes \
  --attention_only {attention_only} --bias_only {bias_only} --static_lm_head {static_lm_head} --static_embedding {static_embedding} \
  --randomly_initialize {randomly_initialize} 
'''
    return cmd


def main(
    output_dir,
    task_name,
    few_shot_type="prompt", # finetune or prompt
    seed=0,
    model_name_or_path="roberta-base",
    data_dir="text_classification/data/original",
    learning_rate=None,
    clipping_mode="ghost",
    clipping_fn="automatic",
    clipping_style="all-layer",
    non_private="no",
    target_epsilon=8,
    attention_only="no",
    bias_only="no",
    static_lm_head="no",
    static_embedding="no",
    physical_batch_size =40,
    eval_steps=10,
    randomly_initialize="no",
    batch_size=None,
    num_train_epochs=None,
):
    command = _get_command(
        output_dir=output_dir,
        task_name=task_name,
        model_name_or_path=model_name_or_path,
        data_dir=data_dir,
        learning_rate=learning_rate,
        clipping_mode=clipping_mode,
        clipping_fn=clipping_fn,
        clipping_style=clipping_style,
        non_private=non_private,
        target_epsilon=target_epsilon,
        few_shot_type=few_shot_type,
        seed=seed,
        attention_only=attention_only,
        bias_only=bias_only,
        static_lm_head=static_lm_head,
        static_embedding=static_embedding,
        physical_batch_size = physical_batch_size,
        eval_steps=eval_steps,
        randomly_initialize=randomly_initialize,
        batch_size=batch_size,
        num_train_epochs=num_train_epochs,
    )
    print('Running command:')
    print(command)
    os.system(command)


if __name__ == "__main__":
    fire.Fire(main)
