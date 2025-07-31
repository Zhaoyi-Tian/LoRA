import pathlib
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
BASE_ROOT = pathlib.Path(__file__).resolve().parent
random.seed(1234)

# ====== 是否使用自定义LoRA权重 ======
use_custom_lora = True  # 设置为True则使用自己实现的LoRA权重

if use_custom_lora:
	# 加载原始模型和分词器
	model_name = str(BASE_ROOT / 'Qwen')
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
	# 注入自定义LoRA层
	from utils.LoRA import MyLoraConfig, inject_lora_layers
	lora_config = MyLoraConfig(
		r=8,
		lora_alpha=16,
		target_modules=["q_proj", "k_proj", "v_proj"],
		lora_dropout=0.05
	)
	model = inject_lora_layers(model, lora_config)
	# 加载自定义LoRA权重（断点权重）
	checkpoint_path = BASE_ROOT / "qwen_lora_finetuned" / "checkpoint-3" / "pytorch_model.bin"
	if checkpoint_path.exists():
		print(f"[INFO] 加载自定义LoRA权重: {checkpoint_path}")
		state = torch.load(str(checkpoint_path), map_location="cuda")
		model.load_state_dict(state, strict=False)
	else:
		print(f"[WARN] 未找到自定义LoRA权重，将仅使用原始模型")
	model.eval()
else:
	# 兼容原有peft包训练的LoRA模型
	lora_model_path = BASE_ROOT / 'qwen_lora_merged'
	if lora_model_path.exists():
		model_name = str(lora_model_path)
		print("[INFO] 正在加载peft包训练的LoRA模型: qwen_lora_merged")
	else:
		model_name = str(BASE_ROOT / 'Qwen')
		print("[INFO] 未检测到LoRA模型，加载原始模型: Qwen")
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
	model.eval()
# 设置特殊token
tokenizer.pad_token = tokenizer.eos_token


def generate_response(prompt, max_new_tokens=512, temperature=0.6, top_p=0.95,top_k=20):
	"""
	生成对话响应
	:param top_k:
	:param prompt: 完整的对话提示 (包含system和history)
	:param max_new_tokens: 最大生成token长度
	:param temperature: 温度参数
	:param top_p: top-p采样参数
	"""
	# 编码输入
	inputs = tokenizer(
		prompt,
		return_tensors="pt",
		truncation=True,
		max_length=1024,
		padding=True
	).to(model.device)

	# 生成响应
	with torch.no_grad():
		outputs = model.generate(
			**inputs,
			max_new_tokens=max_new_tokens,
			do_sample=True,
			temperature=temperature,
			top_p=top_p,
			top_k=top_k,
			eos_token_id=tokenizer.eos_token_id,
			pad_token_id=tokenizer.pad_token_id,
		)

	# 解码并只取生成的响应部分
	input_length = inputs["input_ids"].shape[1]
	generated_ids = outputs[:, input_length:]
	response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
	# 只保留第一个助手回复，遇到下一个 user/assistant 标签时截断
	import re
	response = re.split(r"\nuser:|\nassistant:", response)[0].strip()
	return response


def chat(system_prompt, chat_history=None):
	"""
	多轮对话流
	:param system_prompt: 系统提示
	:param chat_history: 对话历史 [(用户内容, 助手内容), ...]
	"""
	# 构建基础prompt
	prompt = f"system: {system_prompt}\n"

	# 添加历史对话
	if chat_history:
		for user_turn, assistant_turn in chat_history:
			prompt += f"user: {user_turn}\nassistant: {assistant_turn}\n"

	# 当前对话循环
	while True:
		user_input = input("User: ")
		if user_input.lower() in ["exit", "quit"]:
			break

		# 添加当前用户输入到prompt
		current_prompt = prompt + f"user: {user_input}\nassistant: "

		# 生成助手响应
		response = generate_response(current_prompt)
		print(f"Assistant: {response}\n")

		# 更新历史记录（如果需要继续对话）
		chat_history.append((user_input, response))


# 示例使用
if __name__ == "__main__":
	# 定义系统提示 (根据您的应用场景修改)
	system_prompt = "你是一位专业心理咨询师，用温暖、支持性的语言与用户交流。"

	# 初始化对话历史
	history = []

	# 启动对话
	print("系统启动...输入 'exit' 结束对话")
	chat(system_prompt, history)