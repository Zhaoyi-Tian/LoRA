# LoRA

大模型:从基础到实战2000行作业之二，用心理咨询数据LoRA微调Qwen3-1.7B，共579行

```plaintext
MindLoRA/                         # 项目根目录
    ├── data/                     # 数据相关（模型训练等用到的数据存储）
    ├── Qwen/                     # 可能存放 Qwen 模型相关基础文件（如原始模型权重等）
    ├── qwen_lora_finetuned/      # 存放 LoRA 微调后的相关产物（如权重等）
    ├── qwen_lora_merged/         # 存放 LoRA 合并后的相关内容（如合并后的模型权重）
    ├── .gitignore                # Git 忽略规则文件，指定哪些文件/目录不纳入版本控制 
    ├── infer.py                  # 推理相关脚本（用于加载模型进行推理预测等）-134行
    ├── log.txt                   # 日志文件，记录训练、推理等过程中的日志信息 
    ├── LoRA.py                   # LoRA 相关核心实现代码（如 LoRA 模块定义等）-119行
    ├── README.md                 # 项目说明文档
    └── train.py                  # 训练脚本（用于启动模型训练流程）-326行
```
