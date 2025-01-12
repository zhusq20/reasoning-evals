vllm serve /workspace/HuggingFace-Download-Accelerator/qwen25/models--Qwen--Qwen2.5-7B-Instruct

# 2.在simple-evals的父目录运行:
python -m simple-evals.simple_evals --model qwen2.5-7b --debug


'''
simple_evals.py需要修改的地方
line36: models = 后面加新的模型
line113: grading_sampler = models["qwen2.5-7b"]
line114: equality_checker = models["qwen2.5-7b"]改成新的模型名称

'''