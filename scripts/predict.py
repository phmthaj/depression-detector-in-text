import os
import sys
import yaml
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Add root directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import clean_text_func, get_queries
from src.model_utils import load_trained_peft_model

# ==========================================
# 21 CLINICAL SYMPTOM LABELS (ENGLISH ONLY)
# Must match the order of get_queries()
# ==========================================
SYMPTOM_LABELS = [
    "Sadness", 
    "Hopelessness",
    "Past Failure",
    "Anhedonia",
    "Guilt / Self-blame",
    "Feeling Punished",
    "Self-dislike",
    "Self-critical",
    "Suicidal / Self-harm Thoughts",
    "Crying Frequently",
    "Agitation",
    "Social Withdrawal",
    "Indecisiveness",
    "Worthlessness",
    "Loss of Energy",
    "Sleep Changes (Insomnia/Hypersomnia)",
    "Irritability",
    "Appetite Changes",
    "Difficulty Concentrating",
    "Chronic Fatigue",
    "Loss of Libido"
]

def main():
    # 1. Load configuration
    config_path = os.path.join("configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Initializing inference system on: {device.type.upper()}...")

    # 2. Initialize Tokenizer & Queries
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['base_model_name'])
    queries = get_queries()

    # Safety check
    if len(queries) != len(SYMPTOM_LABELS):
        raise ValueError("The number of SYMPTOM_LABELS does not match the number of queries!")

    # 3. Load Fine-tuned LoRA Model
    model_path = cfg['training']['model_save_dir']
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}. Please run train.py first!")
        return
        
    model = load_trained_peft_model(cfg, model_path, device)
    model.eval() 
    print("[INFO] System is ready!\n" + "="*60)

    # 4. Interactive CLI Loop
    print("🤖 ENTER TEXT FOR PSYCHOLOGICAL PROFILE ANALYSIS (Type 'exit' to quit)")
    
    while True:
        try:
            user_input = input("\n📝 You: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue

            clean_text = clean_text_func(user_input)
            detected_symptoms = []
            
            # Scan text against 21 clinical queries
            with torch.no_grad():
                for idx, query in enumerate(queries):
                    encoding = tokenizer(
                        query, 
                        clean_text,
                        max_length=cfg['model']['max_seq_length'],
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    input_ids = encoding['input_ids'].to(device)
                    attention_mask = encoding['attention_mask'].to(device)
                    
                    outputs = model(input_ids, attention_mask=attention_mask)
                    probs = F.softmax(outputs.logits, dim=1)
                    confidence = probs[0][1].item() 
                    
                    # If probability > 50%
                    if confidence > 0.5:
                        symptom_name = SYMPTOM_LABELS[idx]
                        detected_symptoms.append((symptom_name, confidence))

            # Print Results
            print("\n📊 CLINICAL ANALYSIS RESULTS:")
            if not detected_symptoms:
                print("   ✅ Healthy: No abnormal symptoms detected.")
            else:
                print(f"   ⚠️ Detected {len(detected_symptoms)} symptoms requiring attention:")
                # Sort by confidence (descending)
                detected_symptoms.sort(key=lambda x: x[1], reverse=True)
                for symptom, conf in detected_symptoms:
                    # ASCII Progress Bar
                    bar_length = int(conf * 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    print(f"      - {symptom:<36} |{bar}| {conf*100:>5.1f}%")
                    
        except KeyboardInterrupt:
            print("\nForce quit. Goodbye!")
            break

if __name__ == "__main__":
    main()