from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained("/home/dengh/workspace/relational-casemarking-learning/mistral/runs/full-heuristic-run/full-heuristic-run/checkpoint-1000")
tokenizer = AutoTokenizer.from_pretrained("/home/dengh/workspace/relational-casemarking-learning/mistral/runs/full-heuristic-run/full-heuristic-run/checkpoint-1000")
tokenizer.pad_token = tokenizer.eos_token
# 加载验证集（来自 B）
dataset = load_dataset("json", data_files="/home/dengh/workspace/relational-casemarking-learning/data/perturbed/rule/train/rule.validation.jsonl")["train"]

# 分词
def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# 设置评估参数
training_args = TrainingArguments(
    per_device_eval_batch_size=2,
    output_dir="./eval_output",
    do_train=False,
    do_eval=True,
    report_to=[],
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 创建评估器
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 执行评估
metrics = trainer.evaluate()
print(metrics)
