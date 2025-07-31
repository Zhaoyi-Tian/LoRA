from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayerWrapper(nn.Module):
	"""
	将任意线性层转换为 LoRA 适配层
	参数:
		module: 要适配的原始层 (nn.Linear 或 Conv2D)
		r: LoRA 的秩
		alpha: 缩放因子
		dropout: LoRA Dropout 概率
	"""
	def __init__(self, module, r=8, alpha=16, dropout=0.05):
		super().__init__()
		self.module = module
		self.r = r
		self.alpha = alpha
		self.scaling = alpha / r  # 预计算缩放因子
		# 冻结原始层参数
		for param in module.parameters():
			param.requires_grad = False

		out_features, in_features = module.weight.shape
		self.lora_A = nn.Parameter(torch.zeros(r, in_features))
		self.lora_B = nn.Parameter(torch.zeros(out_features, r))
		nn.init.normal_(self.lora_A, std=0.01)
		nn.init.normal_(self.lora_B, std=0.01)
		self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

	def forward(self, input):
		original_output = self.module(input)
		x = self.dropout(input)
		lora_A = self.lora_A.to(x.dtype)
		lora_B = self.lora_B.to(x.dtype)
		x = x @ lora_A.T  # [*, in_dim] -> [*, r]
		x = x @ lora_B.T  # [*, r] -> [*, out_dim]
		return original_output + self.scaling * x

	def merge(self):
		"""合并 LoRA 权重到原始层中"""
		with torch.no_grad():
			# 计算权重增量: ΔW = B@A * scaling
			delta_weight = self.lora_B @ self.lora_A * self.scaling

			# 添加到原始权重
			if isinstance(self.module, nn.Linear):
				self.module.weight += delta_weight
			elif isinstance(self.module, nn.Conv2d):
				# 将卷积核权重视为 [out_ch, in_ch, h, w]
				self.module.weight += delta_weight.view_as(self.module.weight)

@dataclass
class MyLoraConfig:
	r: int = 8
	lora_alpha: int = 16
	target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"])
	lora_dropout: float = 0.05
	bias: str = "none"
	task_type: str = "CAUSAL_LM"


def inject_lora_layers(model, config):
	"""
	递归替换目标模块为 LoRA 适配层
	参数:
		model: 要修改的模型
		config: 包含设置的字典 (r, alpha, lora_dropout, target_modules)
	"""

	def wrap_module(name, child):
		# 检查是否目标模块
		if any(pattern in name for pattern in config.target_modules):
			return LoRALayerWrapper(
				child,
				r=config.r,
				alpha=config.lora_alpha,
				dropout=config.lora_dropout
			)
		return child

	# 递归遍历模型
	def traverse(module, parent_name=""):
		for name, child in module.named_children():
			# 生成完整路径名
			full_name = f"{parent_name}.{name}" if parent_name else name

			# 检查是否直接替换
			wrapped_child = wrap_module(full_name, child)
			if wrapped_child is not child:
				setattr(module, name, wrapped_child)
			else:
				# 递归遍历子模块
				traverse(child, full_name)

	traverse(model)
	return model


def get_lora_optimizer(model, base_lr, weight_decay=1e-4):
	"""为 LoRA 参数创建优化器"""
	# 仅选择 LoRA 参数
	lora_params = []
	for name, param in model.named_parameters():
		if "lora_" in name:  # 只训练 LoRA_A 和 LoRA_B
			param.requires_grad = True
			lora_params.append(param)

	print(f"训练参数数量: {len(lora_params)}/{sum(p.numel() for p in lora_params)}")

	# 创建优化器
	return torch.optim.AdamW(
		lora_params,
		lr=base_lr,
		weight_decay=weight_decay
	)
