import os
import re
import torch
import pandas as pd
import contractions
from tqdm import tqdm
from torch.utils.data import Dataset

# Kích hoạt thanh tiến trình cho Pandas
tqdm.pandas()

# ==========================================
# 1. HÀM TIỀN XỬ LÝ VĂN BẢN (NORMAL CLEANING)
# ==========================================
def clean_text_func(text):
    """
    Làm sạch văn bản cơ bản:
    - Sửa lỗi viết tắt (I'm -> I am)
    - Giữ lại dấu chấm thập phân và khoảng cách số
    - Xóa các ký tự đặc biệt, đưa về chữ thường.
    """
    if pd.isna(text): 
        return ""
        
    text = str(text)
    text = contractions.fix(text).lower()
    text = re.sub(r'(\d+)\.(\d+)', r'\1<dot>\2', text)
    text = re.sub(r'(\d+)-(\d+)', r'\1 to \2', text)
    text = re.sub(r'[^a-z0-9\s<dot>]', '', text)
    text = text.replace("<dot>", ".")
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ==========================================
# 2. DANH SÁCH QUERIES (21 CÂU HỎI LÂM SÀNG)
# ==========================================
def get_queries():
    return [
        "Is the person experiencing feelings of sadness or being blue?",
        "Does the content express a belief that things are hopeless?",
        "Are there feelings related to failure or inadequacy about the past?",
        "Does the text indicate a lack of enjoyment or satisfaction in life?",
        "Does the person express significant feelings of guilt or self-blame?",
        "Are there feelings of being punished or deserving punishment?",
        "Is there an expression of active dislike or disappointment with oneself?",
        "Does the person engage in self-criticism or self-judgment?",
        "Does the text contain thoughts or wishes related to suicide or self-harm?",
        "Are there expressions related to crying or wanting to cry frequently?",
        "Is the person feeling restless, agitated, or unable to sit still?",
        "Does the text mention a loss of general interest in people or activities?",
        "Is there an indication of difficulty in making decisions?",
        "Does the person feel worthless or perceive themselves as having no value?",
        "Is the person reporting a loss of energy or general tiredness?",
        "Are there signs of changes in sleeping habits (insomnia or hypersomnia)?",
        "Is the person expressing increased irritability or frustration?",
        "Are there reported changes in appetite (eating more or less)?",
        "Is the person experiencing difficulty concentrating or thinking clearly?",
        "Does the content show feelings of tiredness or chronic fatigue?",
        "Is the person expressing a loss of interest in sex or sexual activity?"
    ]


# ==========================================
# 3. DATASET CLASS CHO PYTORCH
# ==========================================
class MentalHealthDataset(Dataset):
    def __init__(self, df, tokenizer, queries, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.queries = queries
        
        # 1. BẢO VỆ DỮ LIỆU: Loại bỏ các dòng bị thiếu text
        df_clean = df.copy()
        initial_len = len(df_clean)
        df_clean = df_clean.dropna(subset=['text'])
        if len(df_clean) < initial_len:
            print(f"[INFO] Đã xóa {initial_len - len(df_clean)} dòng bị khuyết 'text'.")
        
        # 2. CHUYỂN ĐỔI LABEL
        if 'relevant' in df_clean.columns:
            df_clean['label'] = df_clean['relevant'].apply(lambda x: 1 if x in [True, 'True', 1] else 0)
        elif 'label' not in df_clean.columns:
            raise KeyError("Dataset phải chứa cột 'relevant' hoặc 'label'.")
            
        # 3. LÀM SẠCH VĂN BẢN (Sử dụng progress_apply để hiện thanh tiến trình)
        print(f"[INFO] Đang làm sạch {len(df_clean)} văn bản...")
        df_clean['clean_text'] = df_clean['text'].progress_apply(clean_text_func)
        
        # Lọc bỏ các dòng rỗng sau khi làm sạch
        df_clean = df_clean[df_clean['clean_text'].str.strip().str.len() > 0].reset_index(drop=True)
        print(f"[INFO] Tải thành công {len(df_clean)} mẫu dữ liệu hợp lệ.")
        
        self.texts = df_clean["clean_text"].tolist()
        self.q_idx = df_clean["query"].astype(int).tolist()
        self.labels = df_clean["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        q_id = self.q_idx[idx] - 1  
        query = self.queries[q_id]
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            query, 
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }