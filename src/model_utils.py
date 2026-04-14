import torch
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

def create_peft_model(cfg, device):
    """
    Khởi tạo mô hình và áp dụng kỹ thuật LoRA dựa trên các tham số từ config.yaml.
    """
    print(f"[INFO] Khởi tạo Base Model: {cfg['model']['base_model_name']}")
    
    # 1. Tải mô hình cơ sở cho bài toán Phân loại Văn bản (Sequence Classification)
    # Sử dụng cfg['model'] để lấy tên model và số lượng nhãn (2)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg['model']['base_model_name'],
        num_labels=cfg['model']['num_labels'],
        ignore_mismatched_sizes=True  # Đảm bảo an toàn nếu checkpoint gốc có số nhãn khác
    ).to(device)

    # 2. Thiết lập cấu hình LoRA từ các key trong mục 'lora' của file YAML
    lora_config = LoraConfig(
        r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['lora_alpha'],
        target_modules=cfg['lora']['target_modules'],  # ["query", "key", "value"]
        lora_dropout=cfg['lora']['lora_dropout'],
        bias="none",
        task_type=TaskType.SEQ_CLS,  # Định nghĩa tác vụ là Phân loại văn bản
    )

    # 3. Gắn Adapter LoRA vào mô hình gốc
    model = get_peft_model(base_model, lora_config)
    
    # Hiển thị các tham số huấn luyện (Quan trọng cho nghiên cứu PEFT của Thái)
    print("[INFO] Cấu trúc tham số sau khi áp dụng LoRA:")
    model.print_trainable_parameters()
    
    return model

def load_trained_peft_model(cfg, model_path, device):
    """
    Tải mô hình đã qua huấn luyện (Base Model + LoRA Adapters) để đánh giá (Evaluate).
    """
    print(f"[INFO] Đang tải mô hình đã huấn luyện từ: {model_path}")
    
    # Tải lại mô hình gốc trước
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg['model']['base_model_name'],
        num_labels=cfg['model']['num_labels']
    ).to(device)
    
    # Tải và gắn các trọng số LoRA đã lưu
    model = PeftModel.from_pretrained(base_model, model_path).to(device)
    model.eval()  # Chuyển sang chế độ đánh giá
    
    return model