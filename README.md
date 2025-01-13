# reasoning-evals

## setup

```bash
pip install vllm --no-build-isolation
pip install transformers

# simple-evals
cd human-eval
pip install -e human-eval
pip install openai

# math evaluations
cd ../Qwen2.5-Math/evaluation/latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt

# livecodebench
cd ../../LiveCodeBench
pip install -e .
```

## MMLU, GPQA, HumanEval, MGSM, DROP

### 1.首先启动待评测的LLM
vllm serve /workspace/HuggingFace-Download-Accelerator/qwen25/models--Qwen--Qwen2.5-7B-Instruct

### 2.更改设置

simple_evals.py需要修改的地方:

line 36: models 里面加你要评测的模型

line 113: grading_sampler = models["qwen2.5-7b"]和
line 114: equality_checker = models["qwen2.5-7b"]改成你想用的模型，
原本的代码用的是gpt-4o

line 156: 设置你想评测的数据集


### 3.在simple-evals的父目录(根目录)运行:
python -m simple-evals.simple_evals --model qwen2.5-7b --debug


## AIME, MATH500, GSM8K

<!-- follow the instructions in ./Qwen2.5-Math/evaluation/README.md -->

You can evaluate Qwen2.5/Qwen2-Math-Instruct series model with the following command:
```bash
cd Qwen2.5-Math/evaluation
# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="qwen25-math-cot"

# Qwen2.5-Math-7B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
```

## MMLU-Pro

To run local inference, modify the model name in the following script and execute it:

```bash
cd MMLU-Pro/scripts/examples/
sh eval_llama_2_7b.sh
```

To use the API for inference, modify the API KEY in evaluate_from_api.py script and execute the bash script:

```bash
cd MMLU-Pro/scripts/examples/
sh eval_gpt_4.sh
```

## LiveCodeBench

follow the instructions in ./benchmarks/LiveCodeBench/README.md

You don’t need to follow the README instructions to run this (it’s troublesome and will cause errors). 
```bash
pip install poetry
poetry install
```

You can directly run the evaluation code and then add the missing libraries as needed.

## ARC-AGI

in progress

## SWE-bench Verified

in progress