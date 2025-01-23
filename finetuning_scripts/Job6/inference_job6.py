import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel


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
    "/project/swabhas_1457/Group_18/Sharan/repairllama-finetuned-default-job6",
    torch_dtype=torch.float16
)

model.config.pad_token_id = tokenizer.pad_token_id = tokenizer.eos_token_id


generation_config = GenerationConfig(
    num_beams=10,
    early_stopping=True,
)

# Buggy C code prompt: code + prompt
buggy_code = """

int main() {
    int i;
    for (i = 0; i < 5; i++)  
    {
        printf("i = %d\n", i); 
    }
    return 0
}



Solve the error above line by line.
"""


inputs = tokenizer(buggy_code, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
inputs_len = inputs["input_ids"].shape[1]

# Generate output with 10 return sequences and attention_mask
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=1024,
    num_return_sequences=10,
    pad_token_id=model.config.pad_token_id,
    eos_token_id=model.config.eos_token_id,
    generation_config=generation_config
)

# Process output
output_ids = outputs[:, inputs_len:]
output_patches = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

# Print generated patches
for i, patch in enumerate(output_patches):
    print(f"Patch {i+1}:")
    print(patch)
    print("-----------------")
