# Transfer Learning-Based Automated Debugging Across Multiple Programming Languages

## Overview
**RepairCLlama** is a fine-tuned model built on **RepairLLaMA** to address automated program repair (APR) tasks for **C/C++ code**. Leveraging transfer learning, it enhances the debugging capabilities originally designed for Java code, enabling accurate and efficient detection and correction of errors in C/C++ programming. 

The project includes:
- **End-to-End Pipeline** for preprocessing, fine-tuning, and evaluating code.
- Extensive use of **Low-Rank Adaptation (LoRA)** for efficient model fine-tuning.
- Fine-tuned models on curated datasets like **Deepfix** and **TEGCER**.
- Evaluation using metrics such as **CodeBLEU**, **AST Similarity**, and **Semantic Match**.

---

## Introduction
RepairCLlama is designed to bridge the gap in automated debugging for C/C++ code. Transfer learning enables it to build on pre-existing Java-trained models, providing an efficient way to address syntactic and semantic errors in C/C++ without training from scratch.


## Pipeline Overview

### Steps
1. **Data Preprocessing**:
   - Conversion to function-level buggy/corrected pairs.
   - Tokenization using Hugging Face tools.

2. **Fine-Tuning**:
   - LoRA-based parameter-efficient adaptation on C/C++ datasets.

3. **Inference**:
   - Guided debugging through structured prompts.
   - Multiple candidate fixes generated using **Beam Search**.

4. **Evaluation**:
   - Metrics: CodeBLEU, AST Similarity, and Semantic Match.

---



## Fine-tuned Models


| Model Name      | Dataset(s) Used  | Hugging Face Link            |
|-----------------|------------------|------------------------------|
| RepairCLlama-v1 | Best Performing Model(TEGCER) | https://huggingface.co/sharan9/RepairCLlama-v1 |
| RepairCLlama-v2 | TEGCER           | https://huggingface.co/sharan9/RepairCLlama-v2 |
| RepairCLlama-v3 | Deepfix          |  https://huggingface.co/sharan9/RepairCLlama-v3         |
| RepairCLlama-v4 | Combined         |  https://huggingface.co/sharan9/RepairCLlama-v4         |





