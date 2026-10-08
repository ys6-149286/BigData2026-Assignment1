# BigData2026 Assignment 1 — Analysis of SVD for Fine-Tuning LLMs

This repository contains the implementation code modified based on the paper **PiSSA (Principal Singular Values and Singular Vectors Adaptation)** published at NeurIPS 2024 (For full paper information, please refer to "References").  

The code is adapted from the original PiSSA implementation and modified to study how different parts of the singular-value spectrum affect low-rank fine-tuning.

Original work:

- Paper: [PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/db36f4d603cc9e3a2a5e10b93e6428f2-Abstract-Conference.html)
- Official code: [MuLabPKU/PiSSA](https://github.com/MuLabPKU/PiSSA)

## File Description

Completed assignment files for submission are in [`submission/`](submission/), ordered by question number:

| Questions | File |
| --- | --- |
| Q1–Q2 | [q1_q2_notebook.ipynb](submission/q1_q2_notebook.ipynb) |
| Q1–Q3 (T4) | [q1_q3_finetune_t4.ipynb](submission/q1_q3_finetune_t4.ipynb) |
| Q3 plotting notebook | [q3_barchart_plot.ipynb](submission/q3_barchart_plot.ipynb) |
| Q3 accuracy chart | [q3_accuracy.png](submission/q3_accuracy.png) |
| Q4 | [q4_finetune_kaggle.ipynb](submission/q4_finetune_kaggle.ipynb) |

The original reference notebooks `finetune_q1-q3.ipynb` and `finetune_q4.ipynb` remain in the repository root.

## Code Description

This code fine-tunes `meta-llama/Llama-3.2-1B` on `MetaMathQA` and evaluates it on two datasets: `GSM8K` and `MATH`.

### 1. Configuration

The configuration cell defines the model, datasets, training hyperparameters, evaluation settings, output directory, and PiSSA runs.

Important settings include:

```python
MODEL_ID = "meta-llama/Llama-3.2-1B"
METAMATH_TRAIN_LIMIT = 25000
MAX_LENGTH = 512
EPOCHS = 1
LEARNING_RATE = 2e-5
TARGET_GLOBAL_BATCH = 128
```

### 2. Settings for SVD

- `rank`: the target rank used for truncated SVD
- `component`: the selected range of singular values, corresponding to the settings in Question 3
- `RUN_FRESH_BASELINE`: used to evaluate the original LLM without fine-tuning


```python

"""
`MANUAL_RUNS` is used to configure the SVD settings for fine-tuning the LLM.
   - 'rank': the target rank used for truncated SVD
   - 'component': the selected range of singular values, corresponding to the settings in Question 3
"""
MANUAL_RUNS = [
   {'rank': 16, 'component': 'default'},
]

"""
`RUN_FRESH_BASELINE` is used to evaluate the original LLM without fine-tuning.
   - True: runs the original LLM as the baseline.
   - False: skips the baseline evaluation.
"""
RUN_FRESH_BASELINE = True
```

### 3. Dataset Preparation

`load_metamathqa_train_like_pissa()`
- Loads the MetaMathQA training set.
- Converts the data into the instruction/response format used for fine-tuning.

`load_pissa_metamath_eval()`
- Loads `fxmeng/pissa-dataset`.
- Separates the evaluation data into GSM8K and MATH tasks.

`tokenize_supervised_example()`
- Tokenizes each instruction/response pair.
- Masks prompt tokens so that training loss is computed only on the response.

`ResponseOnlyCollator`
- Pads examples into training batches.

### 4. Model Initialization

`PiSSALinear`
- Custom linear layer containing the frozen residual weight and trainable `pissa_A` / `pissa_B`.

`pissa_from_linear_exact()`
- Computes the SVD of a pretrained linear layer.
- Selects the requested singular-value region.
- Initializes:

```text
A = U_r sqrt(S_r)
B = sqrt(S_r) V_r^T
```

and stores the remaining weight as:

```text
W_residual = W - AB
```

`inject_pissa()`
- Replaces the selected linear projection modules (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, and down_proj) in each Llama Transformer layer with `PiSSALinear`.

The target projections are:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

`merge_pissa_and_unload()`
- Merges the trained update back into the model:

```text
W_final = W_residual + AB
```

- Converts the custom PiSSA layers back to normal linear layers for evaluation and saving.

### 5. Training

`build_optimizer_and_scheduler()`
- Creates AdamW and the warmup/cosine learning-rate schedule.

`train_one_task()`
- Runs one epoch of fine-tuning.
- Uses FP16 mixed precision and gradient accumulation.
- Only the PiSSA `A` and `B` matrices are trainable.

### 6. Evaluation

Compute "accuracy" for evaluation.

`evaluate_generation()`
- Generates answers and computes correctness.

`evaluate_task()` / `evaluate_all_tasks()`
- Evaluates the model on GSM8K and MATH.

The current repository uses a modified parser that accepts outputs such as:

```text
The answer is: 42
Answer: 42
#### 42
\boxed{42}
```

For MATH, answers are first normalized using the PiSSA evaluation method, then a simple numeric comparison is used as a fallback for equivalent values such as 0.5 and 1/2.

### 7. Experiment Runner

`run_fresh_baseline()`
- Evaluates the original pretrained Llama model.

`run_pissa_experiment()`
- Loads a fresh model.
- Injects PiSSA.
- Trains the selected rank/component.
- Merges the adapter.
- Evaluates the performance on GSM8K and MATH.
- Saves results and model outputs.

`build_run_plan()`
- Builds the list of experiments from `MANUAL_RUNS`.

Results are written to:

```text
/kaggle/working/pissa_benchmark
```

## How to Run ```finetune_q1-q3.ipynb``` on Kaggle

### 1. Enable GPU

Open the notebook in Kaggle and enable a GPU accelerator.  
Two T4 GPUs can be used through `DataParallel`, but the code also runs with one CUDA GPU.

### 2. Add Hugging Face Token

`Llama-3.2-1B` is a gated model.

1. Request access to `meta-llama/Llama-3.2-1B` on Hugging Face.
2. Create a Hugging Face access token.
3. In Kaggle, add the token as a secret named:

```text
HF_TOKEN
```

### 3. Model Settings

Edit `MANUAL_RUNS` before training.

`MANUAL_RUNS` is used to configure the SVD settings for fine-tuning the LLM.
   - 'rank': the target rank used for truncated SVD
   - 'component': the selected range of singular values, corresponding to the settings in Question 3
`RUN_FRESH_BASELINE` is used to evaluate the original LLM without fine-tuning.
   - True: runs the original LLM as the baseline
   - False: skips the baseline evaluation

Example:

```python
MANUAL_RUNS = [
   {'rank': 16, 'component': 'default'},
]

RUN_FRESH_BASELINE = True
```

### 4. Run the Main Pipeline

Run the notebook cells in order on Kaggle.

### 5. Output

Training results will be saved in `/kaggle/working/Assignment1/benchmark_results.csv`, 
including accuracy performance on both datasets.

Under the directory `/kaggle/working/Assignment1/metamath_pissa_{YOUR_COMPONENT_NAME}_r{YOUR_RANK}_full_model/`, you will find files `model.safetensors`, `config.json`, and `generation_config.json`, which can be used to load trained models if needed.


## How to Run ```finetune_q4.ipynb``` on Kaggle

### 1. Add Hugging Face Token

You can reuse your HF_TOKEN created in finetune_q1-q3.ipynb by simply enabling it from "Secrets".

### 2. Running the code

The code for Question 4 is separate from the main fine-tuning experiment, since it's independent from the datasets or the settings above.

## Acknowledgement

This project is adapted from the official [PiSSA](https://github.com/MuLabPKU/PiSSA) implementation by Meng et al. The implementation was modified for alternative singular-subspace initialization, custom experiment settings, and robust GSM8K/MATH evaluation.

## References

1. Meng, F., Wang, Z., & Zhang, M. (2024). **PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models.** *Advances in Neural Information Processing Systems (NeurIPS 2024)*.  
   https://arxiv.org/abs/2404.02948

2. MuLabPKU. **PiSSA: Official Implementation.**  
   https://github.com/MuLabPKU/PiSSA

