
# Finetuning Experiment Setting:

# Job Id: 27807305	
# Dataset: Tegcer	
# Samples: 8000	
# Epochs: 5	
# Max Steps: 2000	
# Learning Rate: 5.00E-04	
# Batch Size: 4	
# Optimizer: Adam(default)
# Training Loss: 0.4923	
# Evaluation Loss: 0.5383
# max_length: 512	
# Model out Folder: repairllama-finetuned-new-epoch-job

from transformers import Trainer, AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers import BitsAndBytesConfig
import os

# Define the dataset class
class BuggyCodeDatasetCSV(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
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

# Load the Tegcer dataset
file_path = "/project/swabhas_1457/Group_18/Sharan/Dataset/Tegcer_ProblemID_buggyCode_correctedCode.csv"
dataset = pd.read_csv(file_path)


print(f"Total samples in dataset: {len(dataset)}")

# Take the first 8000 samples
sampled_dataset = dataset.head(8000)

print(f"Samples used for training: {len(sampled_dataset)}")

# Prepare dataset with buggy and fixed code
processed_data = sampled_dataset[['buggyCode', 'correctedCode']]
train_data = processed_data.sample(frac=0.8, random_state=42)
valid_data = processed_data.drop(train_data.index)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(valid_data)}")


print("\nSample buggy and corrected code:")
print(processed_data.head())

# Initialize tokenizer and model
base_model_path = "codellama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# Load base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    ),
    device_map="auto"
)

# Load RepairLLaMA LoRA weights
repairllama_lora_path = "/project/swabhas_1457/Group_18/Sharan/repairllama-lora"
model = PeftModel.from_pretrained(
    base_model,
    repairllama_lora_path,
    torch_dtype=torch.float16
)

# Create datasets
train_dataset = BuggyCodeDatasetCSV(train_data, tokenizer)
valid_dataset = BuggyCodeDatasetCSV(valid_data, tokenizer)

# Configure QLoRA for full model finetuning (example configuration)
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  
    lora_dropout=0.1,
    bias="all",
    task_type="CAUSAL_LM"
)


model = get_peft_model(model, lora_config)


model.config.pad_token = tokenizer.pad_token = tokenizer.unk_token

output_dir = "/project/swabhas_1457/Group_18/Sharan/repairllama-finetuned-new-epoch-job2"
os.makedirs(output_dir, exist_ok=True)

# Training arguments with increased steps for thorough finetuning
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    logging_dir="/project/swabhas_1457/Group_18/Sharan/logs",
    logging_steps=100,
    save_steps=100,
    eval_steps=100,
    learning_rate=5e-4,
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

# Fine-tune the RepairLLaMA model
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model()
tokenizer.save_pretrained(output_dir)

print("Finetuning completed and model saved.")