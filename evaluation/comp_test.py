from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F

# Load model and tokenizer
checkpoint_path = "/home/dengh/workspace/relational-casemarking-learning/mistral/runs/full-rule-run/full-rule-run/checkpoint-1000"
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token

model.eval()

def calculate_sentence_loss(sentence: str):
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
    
    return loss

# 两个句子
sentence_a = "I 🄰 eat the bread 🄿."
sentence_b = "I eat the bread."
sentence_c = "I 🄰 eat the bread."
sentence_d = "I eat the bread 🄿."

loss_a = calculate_sentence_loss(sentence_a)
loss_b = calculate_sentence_loss(sentence_b)
loss_c = calculate_sentence_loss(sentence_c)
loss_d = calculate_sentence_loss(sentence_d)

print(f"Loss A: {loss_a:.4f}")
print(f"Loss B: {loss_b:.4f}")
print(f"Loss C: {loss_c:.4f}")
print(f"Loss D: {loss_d:.4f}")
