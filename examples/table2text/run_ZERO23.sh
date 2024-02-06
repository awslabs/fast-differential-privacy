#!/bin/bash

data_dir=${1:-"table2text/data/prefix-tuning"}
output_dir=${2:-"table2text/data/output"}
task_mode=${3:-"e2e"}
model_name_or_path=${4:-"gpt2"} # One of "distilgpt2", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
target_epsilon=${5:-8}
clipping_fn=${6:-"automatic"}
clipping_mode=${7:-"MixOpt"}
clipping_style=${8:-"layer-wise"}
bias_only=${9:-"no"}
non_private=${10:-"no"}
physical_batch_size=${11:-4}
learning_rate=${12:-0.001}
batch_size=${13:-1024}
attention_only=${14:-"no"}
static_lm_head=${15:-"no"}
static_embedding=${16:-"no"}
num_GPUs=${17:-8}
deepspeed_config=${18:-"table2text/gpt_config_stage123.json"}

if [[ ${task_mode} == "e2e" ]]; then
  data_dir="${data_dir}/data/e2e_data"
  target_delta=8e-6
  num_train_epochs=10
  max_seq_len=100
else
  if [[ ${task_mode} == "dart" ]]; then
    target_delta=1e-5
    data_dir="${data_dir}/data/dart"
    num_train_epochs=15 # Approximately same number of updates.
    learning_rate=5e-4  # Lower learning rate for stability in large models.
    max_seq_len=120
  else
    echo "Unknown task: ${task_mode}"
    exit 1
  fi
fi

gradient_accumulation_steps=$((${batch_size} / ${physical_batch_size} / ${num_GPUs}))

# Arguments in the last two lines are the most important.
deepspeed table2text/run_language_modeling_ZERO23.py --deepspeed_config ${deepspeed_config} \
  --output_dir ${output_dir} --overwrite_output_dir \
  --task_mode ${task_mode} \
  --model_name_or_path ${model_name_or_path} \
  --tokenizer_name ${model_name_or_path} \
  --do_train --do_eval \
  --line_by_line \
  --save_steps -1 --save_total_limit 1 --save_at_last no \
  --logging_dir ${output_dir} --logging_steps -1 \
  --seed 2 \
  --dataloader_num_workers 2 \
  --eval_steps 100 --eval_epochs 999 --max_eval_batches 100 --evaluation_strategy epoch --evaluate_before_training "no" \
  --evaluate_during_training "no" --per_device_eval_batch_size 10 --skip_generation "yes" \
  --max_generations 9223372036854775807 --max_generations_train 10 --max_generations_valid 9223372036854775807 \
  --max_train_examples 9223372036854775807 --max_valid_examples 9223372036854775807 --max_eval_examples 9223372036854775807 \
  --data_folder ${data_dir} --max_seq_len ${max_seq_len} --format_mode cat \
  --per_example_max_grad_norm 0.1 --target_delta ${target_delta} --target_epsilon ${target_epsilon} \
  --learning_rate ${learning_rate} --lr_decay "no" --num_train_epochs ${num_train_epochs} --per_device_train_batch_size ${physical_batch_size} --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --attention_only ${attention_only} --bias_only ${bias_only} --static_lm_head ${static_lm_head} --static_embedding ${static_embedding} \
  --non_private ${non_private} \
  --clipping_mode "${clipping_mode}" --clipping_fn "${clipping_fn}" --clipping_style "${clipping_style}" 
