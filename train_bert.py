import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import joblib

# Load your dataset
df = pd.read_csv("dataset.csv")  # should contain 'text' and 'label' columns
df = df.dropna()

# Encode labels if they're strings
if df['label'].dtype == object:
    label_map = {label: idx for idx, label in enumerate(df['label'].unique())}
    df['label'] = df['label'].map(label_map)
    joblib.dump(label_map, "label_map.jb")

# Train/Test split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)

# Convert to Hugging Face datasets
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Training Arguments
args = TrainingArguments(
    output_dir='./bert_fakenews',
    evaluation_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir='./logs',
    load_best_model_at_end=True,
    save_total_limit=1,
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

# Train
trainer.train()

# Save model + tokenizer
model.save_pretrained("./bert_fakenews")
tokenizer.save_pretrained("./bert_fakenews")
