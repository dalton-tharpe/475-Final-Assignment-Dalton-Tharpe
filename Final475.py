'''
Dalton Tharpe
04-30-26
CYBV 475
Final, AI vs Real Images
'''
import os
import numpy as np
import torch
from PIL import Image

from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

from huggingface_hub import login
login(input("Input your token here: "))

#-----Load Dataset------
print("Loading Dataset...\n")
dataset = load_dataset("imagefolder", data_dir="dataset")


#-----Load Model and Processor-----
model_name = "google/vit-base-patch16-224"

processor = AutoImageProcessor.from_pretrained(model_name)


#----Safe Label mapping
labels = dataset["train"].features["label"].names
id2label = {i: name for i, name in enumerate(labels)}
label2id = {v: k for k, v in id2label.items()}

print("Label mapping:", id2label)

def preprocess(example):
    image = example["image"].convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    return {
        "pixel_values": inputs["pixel_values"][0],
        "labels": example["label"]
    }

#---remove raw image columnc before training
dataset = dataset.map(preprocess)
dataset = dataset.remove_columns(["image"])
dataset.set_format("torch")


#----Model----
model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)


#-----Training Setup-----
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    learning_rate=5e-5,
    eval_strategy="epoch",
    logging_steps=10,
    save_strategy="no",
    remove_unused_columns=False
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

trainer = Trainer(
    model = model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    data_collator=DefaultDataCollator()
)

#remove extra images



#-----Train model-----
print("Training Model...\n")
trainer.train()

#-----Evaluate-----
print("Evaluating model...\n")
results = trainer.evaluate()
print(f"\nAccuracy: {results['eval_accuracy']:.2f}")


#-----Classify New Folder------
def classify_folder(folder_path):
    print(f"\nClassifying images in: {folder_path}\n")

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, file)

            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs).item()

            label = id2label[pred]
            confidence = probs[0][pred].item()

            print(f"{file}: {label} ({confidence:.2f})")


#----Run Classification-----
classify_folder(input("Input folder to classify: "))