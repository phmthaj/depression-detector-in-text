import os
import sys
import yaml
import torch
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, random_split

# Thêm thư mục gốc vào hệ thống để Python có thể import từ 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import MentalHealthDataset, get_queries
from src.model_utils import create_peft_model
from src.trainer import ModelTrainer

def main():
    # 1. Tải cấu hình
    config_path = os.path.join("configs", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file config tại {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"🚀 Bắt đầu dự án: {cfg.get('project_name', 'Mental Health PEFT')}")
    
    # 2. Thiết lập thiết bị (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Đang sử dụng thiết bị: {device.type.upper()}")

    # 3. Khởi tạo Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['base_model_name'])

    # 4. Chuẩn bị dữ liệu
    train_file_path = os.path.join("data", cfg['data']['train_file'])
    print(f"📂 Đang tải dữ liệu huấn luyện từ: {train_file_path}")
    df = pd.read_csv(train_file_path)
    
    # Tạo Dataset tổng
    full_dataset = MentalHealthDataset(
        df=df, 
        tokenizer=tokenizer, 
        queries=get_queries(), 
        max_len=cfg['model']['max_seq_length']
    )

    # Chia tập Train / Validation
    val_size = int(len(full_dataset) * cfg['data']['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) # Set seed để kết quả tái lập được
    )

    # Khởi tạo DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg['data']['batch_size'] * 2, # Quá trình val không cần gradient nên có thể để batch_size lớn hơn
        shuffle=False
    )

    # 5. Khởi tạo Model & Áp dụng LoRA
    model = create_peft_model(cfg, device)

    # 6. Khởi tạo Trainer và bắt đầu huấn luyện
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device
    )
    
    trainer.train()

if __name__ == "__main__":
    main()