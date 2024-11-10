# LBA-Q2Q 2024
This project leverages the LBA_Q2Q approach to address ambiguities that commonly occur in natural and everyday images during the Visual Question Answering (VQA) process.

## Main Features
- **Clarification Question Generation**: The system generates and answers clarification questions without requiring additional training.
- **Enhanced VQA Accuracy**: By incorporating clarification question and answer pairs, the final VQA accuracy is improved.

## Workflow Description
1. **Input**: The system receives an image, a query, and an ambiguous entity.
2. **Clarification Question Generation**: The model generates yes/no clarification questions aimed at resolving the ambiguity related to the entity.
3. **Answering**: Responses to these questions are provided either by GPT-4o or by human participants.
4. **Final VQA**: The generated clarification question and answer pairs are used as context alongside the original query to conduct the final VQA.


This approach significantly boosts VQA performance by clarifying ambiguous elements within complex, real-world images.

## Setting
To set up the environment, run the following command:

```bash
pip install -r requirements.txt
```

**Note**: Before running the main inference scripts, ensure to execute the files located in the `inference_utils` directory to prepare for passing the evaluation metrics to GPT-4.

## Model
This project utilizes the `llava` model, specifically `llava-hf/llava-v1.6-mistral-7b-hf`.


Ensure that this model is correctly installed and set up in your environment before proceeding.

## Inference
Navigate to the `inference` directory and execute the following scripts sequentially:

```bash
cd inference
python baseline.py
python inference_GPT4_answer.py
python inference_human_answer.py
```

These steps will run the baseline inference and generate answers for evaluation with both GPT-4 and human responses.