import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
from utils import DATA_PATH, MODEL_PATH

# -------------------- Config --------------------
csv_path = os.path.join(DATA_PATH, "training_data_split.csv")
num_labels = 4
epochs = 10
batch_size = 16
lr = 2e-5
max_length = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Dataset --------------------
class AnimacyDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = f"{self.data.iloc[idx]['sentence']} [NP] {self.data.iloc[idx]['np']}"
        label = self.data.iloc[idx]["animacy"]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "label": torch.tensor(label)
        }

# -------------------- Data --------------------
df = pd.read_csv(csv_path)
df = df[df["animacy"].isin(["human", "animal", "inanimate", "event"])]
label_map = {"human": 0, "animal": 1, "inanimate": 2, "event": 3}
df["animacy"] = df["animacy"].map(label_map)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_data = AnimacyDataset(train_df, tokenizer)
test_data = AnimacyDataset(test_df, tokenizer)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# -------------------- Model --------------------
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
model.to(device)

optimizer = AdamW(model.parameters(), lr=lr)

# -------------------- Training --------------------
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")

# -------------------- Save Model --------------------
save_dir = os.path.join(MODEL_PATH, "animacy_bert_model")
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir, safe_serialization=False)
tokenizer.save_pretrained(save_dir)
print(f"Model saved to {save_dir}")

# -------------------- Evaluation --------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["human", "animal", "inanimate", "event"]))
