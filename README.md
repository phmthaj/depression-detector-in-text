# Mental Health Symptom Detection using PEFT (LoRA)

This project fine-tunes a lightweight language model (`all-MiniLM-L6-v2`) using Low-Rank Adaptation (LoRA) to detect 21 clinical mental health symptoms from text inputs.

## 📁 Project Structure

```text
Depression-detection/
├── configs/
│   └── config.yaml          # Hyperparameters and file paths
├── data/
│   ├── dataset2024.csv      # Training dataset
│   └── 2025set.csv          # Evaluation dataset
├── src/
│   ├── data_utils.py        # Text preprocessing & PyTorch Dataset
│   ├── model_utils.py       # Model initialization & LoRA configuration
│   └── trainer.py           # Training loop and evaluation metrics
├── scripts/
│   ├── train.py             # Script to execute model training
│   ├── evaluate.py          # Script to evaluate model on test data
│   └── predict.py           # Interactive inference script
├── requirements.txt         # Python dependencies
└── README.md

1. Install Dependencies
Install the required libraries to set up your environment:
```bash
pip install -r requirements.txt
```

2. Train the Model
Start the fine-tuning process. The script applies LoRA adapters and utilizes early stopping to save the best model based on validation loss.

```bash
python scripts/train.py
```

Outputs: The trained LoRA adapters are saved in data/models/lora_bert_pipeline/.

3. Evaluate the Model (Optional)
Run predictions on the independent test set (2025set.csv) to generate performance metrics (Accuracy, F1-Score, Precision, Recall).

```bash
python scripts/evaluate.py
```

Outputs: Detailed prediction results are exported to data/final_predictions.csv.

4. Run Interactive Inference
Launch the interactive CLI tool. You can input custom text, and the model will analyze it against all 21 clinical queries, returning a real-time symptom profile with confidence scores.

```bash
python scripts/predict.py
```
## 🐳 Docker
Build Image
To pack the environment and dependencies, run the following command in the project's root directory:

```Bash
docker build -t depression_app .
```
1. Run Container (with GPU Support)
To utilize your GPU (e.g., RTX 4050) and sync your code between the host and the container, use the following command:

```bash
docker run --gpus all -it -v ${PWD}:/app depression_app /bin/bash
```

2. Execution Workflow
Once you are inside the container (indicated by the root@...:/app# prompt), you can execute the project scripts in order:

Train the model:

```bash
python scripts/train.py
```
Evaluate the model:

```bash
python scripts/evaluate.py
```
Run inference/prediction:

```bash
python scripts/predict.py
```
