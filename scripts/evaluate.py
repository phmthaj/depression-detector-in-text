import os
import sys
import yaml
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Thêm thư mục gốc vào hệ thống
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import MentalHealthDataset, get_queries
from src.model_utils import load_trained_peft_model

def evaluate():
    # 1. Tải cấu hình
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Bắt đầu quá trình Evaluation trên thiết bị: {device.type.upper()}")

    # 2. Khởi tạo Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['base_model_name'])

    # 3. Tải tập dữ liệu Kiểm thử (Test set)
    test_file_path = os.path.join("data", cfg['data']['test_file'])
    print(f"📂 Đang tải dữ liệu kiểm thử từ: {test_file_path}")
    df_test = pd.read_csv(test_file_path)
    
    # 4. Chuyển đổi qua Dataset & DataLoader
    test_dataset = MentalHealthDataset(
        df=df_test, 
        tokenizer=tokenizer, 
        queries=get_queries(), 
        max_len=cfg['model']['max_seq_length']
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=False
    )

    # 5. Load Model đã huấn luyện
    model_path = cfg['training']['model_save_dir']
    model = load_trained_peft_model(cfg, model_path, device)

    # 6. Chạy quá trình dự đoán
    all_preds = []
    all_labels = []

    print(f"⏳ Đang dự đoán trên {len(test_dataset)} mẫu dữ liệu...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].numpy() # Giữ label ở dạng numpy để tính metrics
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)

    # 7. Tính toán và in báo cáo
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)

    print("\n" + "="*40)
    print(f"✅ KẾT QUẢ ĐÁNH GIÁ (TEST SET)")
    print("="*40)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print("="*40)

    # 8. Lưu kết quả ra CSV
    output_csv = os.path.join("data", "final_predictions.csv")
    
    # Chỉ giữ lại các dòng hợp lệ từ df_test gốc (những dòng có text không bị rỗng)
    # Vì MentalHealthDataset đã lọc các dòng NaN, chúng ta cần gán đúng nhãn cho đúng dòng
    df_results = df_test.dropna(subset=['text']).copy()
    # Lọc tiếp các dòng mà sau khi clean bị rỗng (logic tương tự trong data_utils)
    # Để an toàn nhất, chúng ta lưu text đã xử lý từ Dataset
    
    df_export = pd.DataFrame({
        'query_id': test_dataset.q_idx,
        'clean_text': test_dataset.texts,
        'true_label': all_labels,
        'predicted_label': all_preds
    })
    
    df_export.to_csv(output_csv, index=False)
    print(f"💾 Đã lưu kết quả chi tiết tại: {output_csv}")

if __name__ == "__main__":
    evaluate()