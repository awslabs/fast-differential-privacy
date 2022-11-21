## Reproducing results for text generation

### Requirements

In addition to requirements of the `private-transformers` package, install requirements by running the following from the `table2text` folder of this repo:

```plaintext
pip install -r requirements.txt --no-dependencies
```

### Getting the data

E2E and DART datasets are adapted from \[[Li & Liang, 2021](https://arxiv.org/abs/2101.00190)\] and hosted by \[[Li et al., 2021](https://arxiv.org/abs/2110.05679)\] at [Google drive](https://drive.google.com/file/d/1Re1wyUPtS3IalSsVVJhSg2sn8UNa7DM7/view?usp=sharing). To obtain the data, run
```plaintext
gdown https://drive.google.com/uc?id=1Re1wyUPtS3IalSsVVJhSg2sn8UNa7DM7
unzip prefix-tuning.zip
```
This should produce a `table2text/prefix-tuning/data` subfolder that contains the datasets.

### Running

Use the `run.sh` script in the folder, which runs the `run_language_modeling.py` for the command.

For instance, run the following under the `examples` folder:
```plaintext
bash table2text/run.sh table2text/prefix-tuning <output_dir> "e2e" "gpt2"
```

The script by default uses book-keeping (BK) by [[Differentially Private Optimization on Large Model at Small Cost]](https://arxiv.org/pdf/2210.00038.pdf) for the DP full fine-tuning. Gradient accumulation is used so that larger physical batch size allows faster training at heavier memory burden, but the accuracy is not affected. For E2E/DART, training `gpt2` on one A100 GPU (40GB) takes around 2.5/4 min per epoch.

Arguments (sequentially):
*   `--output_dir`: path to a folder where results will be written

*   `--task_mode`: name of task; one of "e2e" and "dart"

*  `--model_name_or_path`: The pretrained model; one of "distilgpt2", "gpt2", "gpt2-medium", "gpt2-large".

*  `--target_epsilon`: Target privacy spending, default is 8.

*   `--clipping_fn`: Which per-sample gradient clipping function use; one of `automatic` (default, [Bu et al., 2022](https://arxiv.org/pdf/2206.07136.pdf)), `Abadi` [(Abadi et al., 2016)](https://arxiv.org/pdf/1607.00133.pdf) , `global` [(Bu et al., 2021)](https://arxiv.org/pdf/2106.07830.pdf).

*  `--clipping_mode`: Which DP algorithm to implement per-sample gradient clipping; one of `ghost` (default, meaning book-keeping), `MixGhostClip`, `MixOpt`. All three modes are from [Bu et al., 2022](https://arxiv.org/pdf/2210.00038.pdf).
## Reproducing results for text classification

### Requirements

In addition to requirements of the `private-transformers` package, install requirements by running the following from the `text_classification` folder of this repo:

```plaintext
pip install -r requirements.txt --no-dependencies
```

### Getting the data

We adopt the data pipeline by \[[Li et al., 2021](https://arxiv.org/pdf/2110.05679.pdf)\], which is adapted from the excellent work by \[[Gao et al., 2021](https://arxiv.org/pdf/2012.15723.pdf)\]. To obtain the data, run the following:

```plaintext
cd data; bash download_dataset.sh
```

This should produce a `data/original` subfolder that contains the GLUE ([General Language Understanding Evaluation](https://huggingface.co/datasets/glue)) datasets.

### Running

Use the `run_wrapper.py` script in the folder, which runs the `run_classification.py` for the command.

Necessary arguments:

*   `--output_dir`: path to a folder where results will be written
*   `--task_name`: name of task; one of `sst-2`, `qnli`, `qqp`, `mnli`

For instance, run the following under the `examples` folder:

```plaintext
python -m text_classification.run_wrapper --output_dir <output_dir> --task_name <task_name>
```

The script by default uses book-keeping (BK) by [[Differentially Private Optimization on Large Model at Small Cost]](https://arxiv.org/pdf/2210.00038.pdf) for the DP full fine-tuning. Gradient accumulation is used so that larger physical batch size allows faster training at heavier memory burden, but the accuracy is not affected. For SST-2/QNLI/QQP/MNLI, running `roberta-base` on one A100 GPU (40GB) takes around 5/8/37/32 min per epoch.

Additional arguments:

*   `--model_name_or_path`: The pretrained model; one of `distilbert-base-uncased`, `bert-base-uncased`, `bert-large-uncased`, `distilroberta-base`, `roberta-base`, `roberta-large`.

*   `--target_epsilon`: Target privacy spending, default is 8.

*   `--few_shot_type`: Whether to use the generic prompt formatter described in Section 3.2 of our paper. `prompt` is to use, `finetune` is to not use.

*   `--non_private`: Whether to train differentially privately; one of `yes`, `no` (default).

*   `--clipping_mode`: Which DP algorithm to implement per-sample gradient clipping; one of `ghost` (default, meaning book-keeping), `MixGhostClip`, `MixOpt`. All three modes are from [Bu et al., 2022](https://arxiv.org/pdf/2210.00038.pdf).

*   `--clipping_fn`: Which per-sample gradient clipping function use; one of `automatic` (default, [Bu et al., 2022](https://arxiv.org/pdf/2206.07136.pdf)), `Abadi` [(Abadi et al., 2016)](https://arxiv.org/pdf/1607.00133.pdf) , `global` [(Bu et al., 2021)](https://arxiv.org/pdf/2106.07830.pdf).

* `--clipping_style`: Which per-sample gradient clipping style to use; one of `all-layer` (flat clipping), `layer-wise` (each layer is a block, including both weight and bias parameters), `param-wise` (each parameter is a block), or a list of layer names (general block-wise clipping).

*   `--attention_only`: Whether to only train attention layers; one of `yes`, `no` (default).

*   `--bias_only`: Whether to only train bias terms; one of `yes`, `no` (default). If yes, this is implementing [[Differentially Private Bias-Term only
Fine-tuning]](https://arxiv.org/pdf/2210.00036.pdf).

*   `--physical_batch_size` : Physical batch size for gradient accumulation that determines memory and speed, but not accuracy.

*   `--batch_size` : Logical batch size that determines the convergence and accuracy, should be multiple of `physical_batch_size`; default is None.

Note that keeping other training hyperparameter (e.g., number of training epochs, clipping threshold, learning rate) as default,  the script should reproduce the results in \[[Li et al., 2021](https://arxiv.org/pdf/2110.05679.pdf); [Bu et al., 2022](https://arxiv.org/pdf/2206.07136.pdf)\].
*  `--bias_only`: Whether to only train bias terms; one of `yes`, `no` (default). If yes, this is implementing [[Differentially Private Bias-Term only
Fine-tuning]](https://arxiv.org/pdf/2210.00036.pdf).

*  `--non_private`: Whether to train differentially privately; one of `yes`, `no` (default).

*  `--physical_batch_size` : Physical batch size for gradient accumulation that determines memory and speed, but not accuracy.

*  `--batch_size` : Logical batch size that determines the convergence and accuracy, should be multiple of `physical_batch_size`; default is None.

*  `--attention_only`: Whether to only train attention layers; one of `yes`, `no` (default).

Note that keeping other training hyperparameter (e.g., number of training epochs, learning rate) as default,  the script should reproduce the results in \[[Li et al., 2021](https://arxiv.org/pdf/2110.05679.pdf); [Bu et al., 2022](https://arxiv.org/pdf/2206.07136.pdf)\].

### Evaluation

The script automatically evaluates some measures like loss during the training. To evaluate the generations with BLEU, ROGUE, METEOR, CIDEr, NIST, etc., we use the official [e2e-metrics](https://github.com/tuetschek/e2e-metrics) for E2E, and [GEM-metrics](https://github.com/GEM-benchmark/GEM-metrics) for DART.

Specifically for E2E, after installing e2e-metric in the `table2text` folder, run
```bash
cpanm --local-lib=~/perl5 local::lib && eval $(perl -I ~/perl5/lib/perl5/ -Mlocal::lib)
python e2e-metrics/measure_scores.py prefix-tuning/data/e2e_data/clean_references_test.txt ../<output_dir>/generations_model/eval/global_step_00000420.txt
```
