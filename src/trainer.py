import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, cfg, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        
        # Thiết lập Optimizer và Loss function
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=float(cfg['training']['learning_rate'])
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Thiết lập Early Stopping
        self.patience = cfg['training']['patience']
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.save_dir = cfg['training']['model_save_dir']
        
        os.makedirs(self.save_dir, exist_ok=True)

    def validate(self):
        """Hàm đánh giá mô hình trên tập Validation."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)
                total_loss += loss.item()
                
                preds = outputs.logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_loss = total_loss / len(self.val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        
        self.model.train() 
        return avg_loss, acc, f1, precision, recall

    def train(self):
        """Vòng lặp huấn luyện chính."""
        epochs = self.cfg['training']['epochs']
        print(f"\n[INFO] Bắt đầu huấn luyện với {epochs} Epochs trên thiết bị: {self.device.type.upper()}")
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_train_loss = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs} (Train)")
            
            for batch in progress_bar:
                self.optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)
                
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item()
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_train_loss = total_train_loss / len(self.train_loader)
            
            # Đánh giá trên tập Val
            val_loss, val_acc, val_f1, val_prec, val_rec = self.validate()
            
            print(f"\n[Epoch {epoch} Kết quả] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"      Validation Metrics -> Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

            # Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"🌟 Val Loss cải thiện! Lưu mô hình tại: {self.save_dir}")
                self.model.save_pretrained(self.save_dir)
            else:
                self.patience_counter += 1
                print(f"⚠️ Val Loss không giảm. Patience: {self.patience_counter}/{self.patience}")
                if self.patience_counter >= self.patience:
                    print(f"⛔ Dừng huấn luyện sớm (Early Stopping) do không cải thiện sau {self.patience} epochs.")
                    break

        print("\n[INFO] Hoàn tất quá trình huấn luyện!")