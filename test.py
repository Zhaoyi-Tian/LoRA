#source /home/tzy/.virtualenvs/MindLoRA/bin/activate
import pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_ROOT = pathlib.Path(__file__).resolve().parent
checkpoint_dir = str(BASE_ROOT / "qwen_lora_finetuned" / "checkpoint-2380")
base_model_dir = str(BASE_ROOT / "Qwen")  # 原始模型目录

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_dir,
    torch_dtype="auto",
    device_map="auto",
)

# 加载LoRA权重
model = PeftModel.from_pretrained(model, checkpoint_dir)

# 合并LoRA权重
model = model.merge_and_unload()

# 加载并保存分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

# 保存合并后的模型和分词器
output_dir = str(BASE_ROOT / "qwen_lora_merged")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
