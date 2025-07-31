#=========================================================================================
import json
import pathlib
import os
import math
from utils.LoRA import MyLoraConfig, inject_lora_layers, get_lora_optimizer, LoRALayerWrapper

os.environ["TRANSFORMERS_NO_MLX"] = "1"

import sys
sys.modules["mlx"] = None
sys.modules["mlx.core"] = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
#=======================================================================================
# 调peft包
use_peft=True

#=======================================================================================

BASE_ROOT=pathlib.Path(__file__).resolve().parent
model_name=BASE_ROOT/"Qwen"
train_data = []
test_data = []


# ====== 数据加载异常处理 ======
train_json_path = BASE_ROOT / 'data' / 'PsyDTCorpus_train_mulit_turn_packing.json'
try:
    with open(str(train_json_path), 'r', encoding='utf-8') as f:
        new_train_data = json.load(f)
except FileNotFoundError:
    print(f"训练数据文件未找到: {train_json_path}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"训练数据JSON解析失败: {e}")
    exit(1)

for item in new_train_data:
    try:
        messages = item["messages"]
        system_prompt = next(msg["content"] for msg in messages if msg["role"] == "system")
        prompt_parts = [f"system: {system_prompt}"]
        response = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt_parts.append(f"user: {msg['content']}")
            elif msg["role"] == "assistant":
                if msg == messages[-1]:
                    response = msg["content"]
                else:
                    prompt_parts.append(f"assistant: {msg['content']}")
        prompt = "\n".join(prompt_parts)
        train_data.append({"prompt": prompt, "response": response})
    except Exception as e:
        print(f"训练数据处理异常: {e}")

train_dataset = Dataset.from_list(train_data)



# ====== 测试数据加载异常处理 ======
test_json_path = BASE_ROOT / 'data' / 'PsyDTCorpus_test_single_turn_split.json'
try:
    with open(str(test_json_path), 'r', encoding='utf-8') as f:
        new_test_data = json.load(f)
except FileNotFoundError:
    print(f"测试数据文件未找到: {test_json_path}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"测试数据JSON解析失败: {e}")
    exit(1)

for item in new_test_data:
    try:
        messages = item["messages"]
        system_prompt = next(msg["content"] for msg in messages if msg["role"] == "system")
        prompt_parts = [f"system: {system_prompt}"]
        response = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt_parts.append(f"user: {msg['content']}")
            elif msg["role"] == "assistant":
                if msg == messages[-1]:
                    response = msg["content"]
                else:
                    prompt_parts.append(f"assistant: {msg['content']}")
        prompt = "\n".join(prompt_parts)
        test_data.append({"prompt": prompt, "response": response})
    except Exception as e:
        print(f"测试数据处理异常: {e}")

test_dataset = Dataset.from_list(test_data)
for i in range(5):
    print(train_data[i])

def preprocess_function(examples):
    # 处理输入prompt，添加padding
    inputs = tokenizer(
        examples["prompt"],
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt"
    )

    # 处理标签response，添加padding并替换填充部分为-100
    labels = tokenizer(
        examples["response"],
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt"
    )

    # 将填充位置的标签设为-100（忽略损失计算）
    labels["input_ids"] = torch.where(
        labels["input_ids"] == tokenizer.pad_token_id,
        torch.tensor(-100),
        labels["input_ids"]
    )

    # 构建正确的标签：输入部分为-100，回复部分为实际标签
    full_labels = -100 * torch.ones_like(inputs["input_ids"])
    response_length = labels["input_ids"].shape[1]

    # 将response标签拼接到完整标签的末尾
    for i in range(len(full_labels)):
        full_labels[i, -response_length:] = labels["input_ids"][i]

    inputs["labels"] = full_labels
    return inputs


# ====== 模型和分词器加载异常处理 ======
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    print(f"分词器加载失败: {e}")
    exit(1)

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)
tokenizer.pad_token = tokenizer.eos_token

# print("模型模块列表：")
# for name, _ in model.named_modules():
#     if "attn" in name or "query" in name or "key" in name or "value" in name:
#         print(name)  # 只打印可能与注意力相关的模块
#  配置LoRA
if use_peft:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj"],  # Qwen 注意力相关模块
        lora_dropout=0.05,
        bias="none",
        inference_mode=False
    )
    model = get_peft_model(model, peft_config)
else:
    lora_config = MyLoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05
    )
    model = inject_lora_layers(model, lora_config)

# 数据预处理
train_dataset = train_dataset.map(preprocess_function, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset = test_dataset.map(preprocess_function, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 训练设置

training_args = TrainingArguments(
    max_grad_norm=1.0,
    output_dir="./qwen_lora_finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=3e-4,
    logging_steps=10,
    fp16=True,
    push_to_hub=False,
    eval_strategy="epoch",
    save_strategy="epoch",       # 每个epoch结束后保存（新增参数）
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="loss",  # 用损失作为评估指标（更适合语言模型）
    greater_is_better=False , # 损失越小越好
    eval_accumulation_steps=4,  # 增大该值可减少评估时的内存占用
    fp16_full_eval=True        # 启用全评估阶段的混合精度
 )
# 创建自定义优化器和调度器
optimizer = get_lora_optimizer(model, base_lr=training_args.learning_rate)
# 计算总训练步数
total_steps = (len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)) * training_args.num_train_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


# ===================== 断点续训功能 =====================
import glob
def get_latest_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*") )
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]


# ====== 断点检测异常处理 ======
try:
    resume_checkpoint = get_latest_checkpoint(training_args.output_dir)
    if resume_checkpoint:
        print(f"检测到断点: {resume_checkpoint}，将从断点继续训练...")
    else:
        print("未检测到断点，将从头开始训练...")
except Exception as e:
    print(f"断点检测异常: {e}")
    resume_checkpoint = None


if use_peft:
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            optimizers=(optimizer, scheduler),  # 使用自定义优化器
        )
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        model = model.merge_and_unload()
        model.save_pretrained("./qwen_lora_merged")
        tokenizer.save_pretrained("./qwen_lora_merged")
    except Exception as e:
        print(f"训练过程异常: {e}")
        exit(1)
else:
    # PyTorch原生训练流程，支持断点续训
    try:
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        max_grad_norm = 1.0
        start_epoch = 0
        # 检查是否有断点
        checkpoint_path = get_latest_checkpoint("./qwen_lora_finetuned")
        if checkpoint_path:
            print(f"加载断点: {checkpoint_path}")
            try:
                state = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location=device)
                model.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"断点加载失败: {e}")
        model.train()
        for epoch in range(start_epoch, 3):
            total_loss = 0.0
            step = 0
            for batch in train_loader:
                try:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.item()
                    step += 1
                    print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")
                except Exception as e:
                    print(f"训练step异常: {e}")
            avg_loss = total_loss / step if step > 0 else 0
            print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
            # 保存断点
            save_dir = f"./qwen_lora_finetuned/checkpoint-{epoch+1}"
            try:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
                print(f"已保存断点: {save_dir}")
            except Exception as e:
                print(f"断点保存失败: {e}")
        # 合并LoRA权重并保存
        for module in model.modules():
            if isinstance(module, LoRALayerWrapper):
                module.merge()
        try:
            model.save_pretrained("./qwen_lora_merged")
            tokenizer.save_pretrained("./qwen_lora_merged")
        except Exception as e:
            print(f"模型保存失败: {e}")
    except Exception as e:
        print(f"训练主流程异常: {e}")
        exit(1)