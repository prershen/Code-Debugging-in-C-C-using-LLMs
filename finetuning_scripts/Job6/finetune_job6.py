
# Finetuning Experiment Setting:

# Job Id: 28332144	
# Dataset: Tegcer	
# Samples: 10000	
# Epochs: 5	
# Max Steps: 2000	
# Learning Rate: 2.00E-05	
# Batch Size: 4	
# Optimizer: Adam(default)
# Training Loss: 0.309220377	
# Evaluation Loss: 0.2712129653	
# max_length: 1024	
# Model out Folder: repairllama-finetuned-default-job6

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
import pandas as pd
from torch.utils.data import Dataset
import torch
import os

# Define the dataset class
class BuggyCodeDatasetCSV(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=1024):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        buggy_code = sample['buggyCode']
        fixed_code = sample['correctedCode']

        # Tokenize input and labels
        inputs = self.tokenizer(
            buggy_code, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        labels = self.tokenizer(
            fixed_code, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        inputs["labels"] = labels["input_ids"]
        return {key: val.squeeze(0) for key, val in inputs.items()}

# Load the dataset
file_path = "/project/swabhas_1457/Group_18/Sharan/Dataset/Tegcer_ProblemID_buggyCode_correctedCode.csv"
dataset = pd.read_csv(file_path)

print(f"Total samples in dataset: {len(dataset)}")

# Considering first 10,000 samples
sampled_dataset = dataset.head(10000)
print(f"Samples used for training: {len(sampled_dataset)}")

processed_data = sampled_dataset[['buggyCode', 'correctedCode']]
train_data = processed_data.sample(frac=0.8, random_state=42)
valid_data = processed_data.drop(train_data.index)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(valid_data)}")

# sample data
print("\nSample buggy and corrected code:")
print(processed_data.head())

# Base model and LoRA adapter repo IDs
base_model_id = "codellama/CodeLlama-7b-hf"
repairllama_lora_repo = "ASSERT-KTH/RepairLLaMA-IR1-OR1"


tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

# Load base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    ),
    device_map="auto"
)

# Load the LoRA adapter from Hugging Face Hub
model = PeftModel.from_pretrained(
    base_model,
    repairllama_lora_repo,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Create datasets
train_dataset = BuggyCodeDatasetCSV(train_data, tokenizer)
valid_dataset = BuggyCodeDatasetCSV(valid_data, tokenizer)

# Configure LoRA for full model fine-tuning
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="all",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Ensure tokenizer has a pad token
model.config.pad_token = tokenizer.pad_token = tokenizer.unk_token

# Prepare output directories
output_dir = "/project/swabhas_1457/Group_18/Sharan/repairllama-finetuned-default-job6"
logging_dir = "/project/swabhas_1457/Group_18/Sharan/logs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    logging_dir=logging_dir,
    logging_steps=100,
    save_steps=100,
    eval_steps=100,
    learning_rate=2e-5,
    weight_decay=0.1,
    fp16=True,
    gradient_accumulation_steps=4,
    save_total_limit=2,
    load_best_model_at_end=True,
    max_steps=2000,
    lr_scheduler_type="cosine",
    warmup_ratio=0.02
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

# Merge LoRA weights into the base model if needed
model.merge_and_unload()

# Save the fine-tuned model and tokenizer
trainer.save_model()
tokenizer.save_pretrained(output_dir)

print("Finetuning completed and model saved.")
