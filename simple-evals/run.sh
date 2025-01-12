# 1.首先启动待评测的LLM
vllm serve /workspace/HuggingFace-Download-Accelerator/qwen25/models--Qwen--Qwen2.5-7B-Instruct

# 2.更改设置
'''
simple_evals.py需要修改的地方:

line 36: models 里面加你要评测的模型

line 113: grading_sampler = models["qwen2.5-7b"]和
line 114: equality_checker = models["qwen2.5-7b"]改成你想用的模型，
原本的代码用的是gpt-4o

line 156: 设置你想评测的数据集
'''

# 3.在simple-evals的父目录(根目录)运行:
python -m simple-evals.simple_evals --model qwen2.5-7b --debug
