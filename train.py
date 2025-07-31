#=========================================================================================
import json
import pathlib
import os
from LoRA import MyLoraConfig, inject_lora_layers, get_lora_optimizer, LoRALayerWrapper

os.environ["TRANSFORMERS_NO_MLX"] = "1"

import sys
sys.modules["mlx"] = None
sys.modules["mlx.core"] = None

import torch
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

new_train_data=json.load(open(str(BASE_ROOT / 'data' / 'PsyDTCorpus_train_mulit_turn_packing.json')))
for item in new_train_data:
    messages = item["messages"]
    # 提取system提示（固定在开头）
    system_prompt = next(msg["content"] for msg in messages if msg["role"] == "system")
    # 拼接多轮对话（除最后一条assistant回复外，均作为prompt的一部分）
    prompt_parts = [f"system: {system_prompt}"]
    response = ""
    # 遍历消息，收集对话历史和最后一条assistant回复
    for msg in messages:
        if msg["role"] == "user":
            prompt_parts.append(f"user: {msg['content']}")
        elif msg["role"] == "assistant":
            # 最后一条assistant消息作为response，前面的作为prompt
            if msg == messages[-1]:
                response = msg["content"]
            else:
                prompt_parts.append(f"assistant: {msg['content']}")
    # 拼接完整prompt
    prompt = "\n".join(prompt_parts)
    train_data.append({"prompt": prompt, "response": response})

train_dataset = Dataset.from_list(train_data)


new_test_data=json.load(open(str(BASE_ROOT / 'data' / 'PsyDTCorpus_test_single_turn_split.json')))
for item in new_test_data:
    messages = item["messages"]
    # 提取system提示（固定在开头）
    system_prompt = next(msg["content"] for msg in messages if msg["role"] == "system")
    # 拼接多轮对话（除最后一条assistant回复外，均作为prompt的一部分）
    prompt_parts = [f"system: {system_prompt}"]
    response = ""
    # 遍历消息，收集对话历史和最后一条assistant回复
    for msg in messages:
        if msg["role"] == "user":
            prompt_parts.append(f"user: {msg['content']}")
        elif msg["role"] == "assistant":
            # 最后一条assistant消息作为response，前面的作为prompt
            if msg == messages[-1]:
                response = msg["content"]
            else:
                prompt_parts.append(f"assistant: {msg['content']}")
    # 拼接完整prompt
    prompt = "\n".join(prompt_parts)
    test_data.append({"prompt": prompt, "response": response})

test_dataset = Dataset.from_list(test_data)


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

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 新增：配置8bit量化参数（替代原load_in_8bit=True）

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
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

# 创建Trainer并训练
if use_peft:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        optimizers=(optimizer, scheduler),  # 使用自定义优化器
    )
    trainer.train(resume_from_checkpoint=True)
    model = model.merge_and_unload()
    model.save_pretrained("./qwen_lora_merged")
    tokenizer.save_pretrained("./qwen_lora_merged")
else:
    # PyTorch原生训练流程
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    max_grad_norm = 1.0
    model.train()
    for epoch in range(3):
        total_loss = 0.0
        step = 0
        for batch in train_loader:
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
        avg_loss = total_loss / step if step > 0 else 0
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
    # 合并LoRA权重并保存
    for module in model.modules():
        if isinstance(module, LoRALayerWrapper):
            module.merge()
    model.save_pretrained("./qwen_lora_merged")
    tokenizer.save_pretrained("./qwen_lora_merged")