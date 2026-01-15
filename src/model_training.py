"""
Модуль для обучения CLIP модели на датасете одежды
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
from config.paths import PATHS


class CLIPDataset(Dataset):
    """Dataset для CLIP обучения с аугментацией"""
    def __init__(self, dataframe, image_folder, processor, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.image_folder = image_folder
        self.processor = processor
        self.augment = augment
        
        # Аугментация (только для train)
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(10),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_folder, row['image'])
        image = Image.open(img_path).convert('RGB')
        
        # Применяем аугментацию
        if self.transform:
            image = self.transform(image)
        
        text = row['full_text']
        
        return {'image': image, 'text': text}


def collate_fn(batch, processor):
    """Collate функция для батчей"""
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    return inputs


def validate(model, dataloader, device):
    """Валидация модели с loss и score"""
    model.eval()
    total_score = 0
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, return_loss=True)
            
            # Loss
            total_loss += outputs.loss.item()
            
            # Score
            scores = torch.diagonal(outputs.logits_per_image)
            total_score += scores.sum().item()
            count += len(scores)
    
    avg_loss = total_loss / len(dataloader)
    avg_score = total_score / count
    
    return avg_loss, avg_score


def train_clip(model, train_loader, test_loader, epochs=10, lr=5e-6, patience=3, device='cpu'):
    """
    Обучение CLIP модели с LR scheduling и Early Stopping
    
    Args:
        model: CLIP модель
        train_loader: DataLoader для обучения
        test_loader: DataLoader для валидации
        epochs: количество эпох
        lr: learning rate
        patience: количество эпох без улучшения для early stopping
        device: устройство (cpu/cuda)
    
    Returns:
        model: обученная модель
        history: история обучения
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        'train_loss': [],
        'train_score': [],
        'test_loss': [],
        'test_score': [],
        'learning_rate': []
    }
    
    os.makedirs(PATHS['checkpoints'], exist_ok=True)
    
    # Early stopping
    best_score = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_score = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch, return_loss=True)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                scores = torch.diagonal(outputs.logits_per_image)
                batch_score = scores.mean().item()
            
            epoch_loss += loss.item()
            epoch_score += batch_score
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'score': f'{batch_score:.2f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = epoch_loss / len(train_loader)
        avg_train_score = epoch_score / len(train_loader)
        
        # Валидация
        test_loss, test_score = validate(model, test_loader, device)
        current_lr = scheduler.get_last_lr()[0]
        
        history['train_loss'].append(avg_loss)
        history['train_score'].append(avg_train_score)
        history['test_loss'].append(test_loss)
        history['test_score'].append(test_score)
        history['learning_rate'].append(current_lr)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Train Score: {avg_train_score:.2f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Score: {test_score:.2f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Сохраняем ТОЛЬКО лучшую модель
        if test_score > best_score:
            best_score = test_score
            patience_counter = 0
            
            best_path = os.path.join(PATHS['checkpoints'], 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_score': test_score,
                'history': history
            }, best_path)
            print(f"  🏆 Новая лучшая модель! Score: {test_score:.2f}")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping! Лучший Test Score: {best_score:.2f}")
            break
        
        # Обновляем learning rate
        scheduler.step()
        print()
    
    # Загружаем лучшую модель
    best_checkpoint = torch.load(os.path.join(PATHS['checkpoints'], 'best_model.pt'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f"\n✓ Загружена лучшая модель (эпоха {best_checkpoint['epoch']+1}, Score: {best_checkpoint['test_score']:.2f})")
    
    return model, history


def plot_history(history):
    """Визуализация истории обучения"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss (Train & Test)
    axes[0].plot(history['train_loss'], marker='o', linewidth=2, label='Train', color='#FF6B6B')
    axes[0].plot(history['test_loss'], marker='s', linewidth=2, label='Test', color='#FFB6B6')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 2. CLIP Scores
    axes[1].plot(history['train_score'], marker='o', linewidth=2, label='Train', color='#4ECDC4')
    axes[1].plot(history['test_score'], marker='s', linewidth=2, label='Test', color='#95E1D3')
    axes[1].axhline(y=30, color='red', linestyle='--', linewidth=2, label='Target (30)')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('CLIP Score', fontsize=12)
    axes[1].set_title('CLIP Scores', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Learning Rate
    axes[2].plot(history['learning_rate'], marker='o', linewidth=2, color='#A8E6CF')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохраняем график
    plot_path = os.path.join(PATHS['checkpoints'], 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ График сохранен: {plot_path}\n")
    
    plt.show()
    
    # Статистика
    print("="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)
    print(f"Всего эпох: {len(history['train_loss'])}")
    print(f"\nЛучший Test Score: {max(history['test_score']):.2f} (эпоха {history['test_score'].index(max(history['test_score']))+1})")
    print(f"Финальный Train Score: {history['train_score'][-1]:.2f}")
    print(f"Финальный Test Score: {history['test_score'][-1]:.2f}")
    print(f"Финальный Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Финальный Test Loss: {history['test_loss'][-1]:.4f}")
    print(f"\nУлучшение Test Score: {history['test_score'][-1] - history['test_score'][0]:.2f}")
    print("="*60)


if __name__ == "__main__":
    # Загружаем данные
    train_df = pd.read_csv(os.path.join(PATHS['processed_data'], 'split', 'train.csv'))
    test_df = pd.read_csv(os.path.join(PATHS['processed_data'], 'split', 'test.csv'))
    
    # Добавляем full_text если нет
    if 'full_text' not in train_df.columns:
        train_df['full_text'] = train_df['display name'] + '. ' + train_df['description']
        test_df['full_text'] = test_df['display name'] + '. ' + test_df['description']
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Загружаем модель
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # ==================== АУГМЕНТАЦИЯ ====================
    print("="*60)
    print("НАСТРОЙКА АУГМЕНТАЦИИ")
    print("="*60)
    print("Train dataset: С аугментацией ✓")
    print("  - RandomResizedCrop (scale=0.8-1.0)")
    print("  - RandomHorizontalFlip")
    print("  - ColorJitter (brightness, contrast, saturation=0.2)")
    print("  - RandomRotation (±10°)")
    print("Test dataset: Без аугментации")
    print("="*60 + "\n")
    # ======================================================
    
    # Создаем датасеты
    image_folder = os.path.join(PATHS['extracted_data'], 'data')
    
    train_dataset = CLIPDataset(train_df, image_folder, processor, augment=True)
    test_dataset = CLIPDataset(test_df, image_folder, processor, augment=False)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, processor)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=lambda b: collate_fn(b, processor)
    )
    
    # Обучаем
    model, history = train_clip(
        model, 
        train_loader, 
        test_loader, 
        epochs=10, 
        lr=5e-6, 
        patience=3,
        device=device
    )
    
    # Визуализация
    plot_history(history)