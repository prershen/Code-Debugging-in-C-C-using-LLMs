import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# Paths
test_path = "data/Tegcer_Test.csv"
output_path = "inferences_generated_Tegcer_Test.csv"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", trust_remote_code=True)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# Load base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-hf",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    quantization_config=quantization_config,
    device_map="auto"
)

# Load fine-tuned model with LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "repairllama-finetuned-default-job6",
    torch_dtype=torch.float16
)

# Set pad token
model.config.pad_token_id = tokenizer.pad_token_id = tokenizer.eos_token_id

# Generation configuration
generation_config = GenerationConfig(
    num_beams=10,
    early_stopping=True,
)

def generate_corrections(buggy_code, model, tokenizer):
    """
    Generate corrections for a single buggy code sample
    """
    # Prepare input
    inputs = tokenizer(
        buggy_code + "\nSolve the error above line by line.", 
        return_tensors="pt", 
        max_length=1024, 
        truncation=True
    ).to(model.device)
    
    inputs_len = inputs["input_ids"].shape[1]
    
    # Generate output
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        #max_length=1024,
        max_new_tokens=256,
        num_return_sequences=10,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        generation_config=generation_config
    )
    
    # Process output
    output_ids = outputs[:, inputs_len:]
    output_patches = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return output_patches

def process_test_dataset(test_dataset_path, output_path):
    """
    Process the entire test dataset and generate corrections
    """
    # Load test dataset
    test_data = pd.read_csv(test_dataset_path)
    test_data = test_data.head(100)  # take only 100

    # Prepare output list
    all_corrections = []

    # Process each sample in the dataset
    for _, sample in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing Test Dataset"):
        problem_id = sample['ProblemID']
        buggy_code = sample['buggyCode']
        original_correct_code = sample.get('correctedCode', '')  # Optional, in case it's available
        
        # Generate corrections for the buggy code
        corrections = generate_corrections(buggy_code, model, tokenizer)
        
        # Append each correction as a separate row
        for i, correction in enumerate(corrections, 1):
            sample_corrections = {
                'ProblemID': problem_id,
                'buggyCode': buggy_code,
                'originalCorrectCode': original_correct_code,
                'generatedCorrection': correction,
                'correctionNumber': i
            }
            
            all_corrections.append(sample_corrections)

    # Save corrections to output file
    output_df = pd.DataFrame(all_corrections)
    output_df.to_csv(output_path, index=False)

    print(f"Processed {len(test_data)} samples. Corrections saved to {output_path}")

# Process the dataset
if __name__ == "__main__":
    process_test_dataset(test_path, output_path)