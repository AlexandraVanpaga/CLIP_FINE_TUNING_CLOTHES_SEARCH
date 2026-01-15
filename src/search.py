"""
Модуль для поиска товаров по текстовому запросу
"""
import torch
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from config.paths import PATHS


def load_embeddings(embeddings_path):
    """Загрузка сохраненных embeddings"""
    data = np.load(embeddings_path, allow_pickle=True)
    
    # Загружаем dataframe из CSV
    train_df = pd.read_csv(os.path.join(PATHS['processed_data'], 'split', 'train.csv'))
    test_df = pd.read_csv(os.path.join(PATHS['processed_data'], 'split', 'test.csv'))
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    return {
        'embeddings': data['embeddings'],
        'image_names': list(data['image_names']),
        'dataframe': full_df
    }

def search_products(query, model, processor, image_embeddings, image_names, dataframe, top_k=5, device='cpu'):
    """
    Поиск товаров по текстовому запросу
    
    Args:
        query: текстовый запрос
        model: дообученная CLIP модель
        processor: CLIP processor
        image_embeddings: numpy array с embeddings изображений
        image_names: список названий файлов
        dataframe: DataFrame с данными
        top_k: количество результатов
        device: устройство
    
    Returns:
        list of dict с результатами
    """
    model.eval()
    
    # Кодируем текстовый запрос
    with torch.no_grad():
        inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
        text_features = model.get_text_features(input_ids=inputs['input_ids'].to(device))
        
        # Нормализуем
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()
    
    # Вычисляем косинусное сходство
    similarities = np.dot(image_embeddings, text_features.T).squeeze()
    
    # Топ-K результатов
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Формируем результаты
    results = []
    for idx in top_indices:
        img_name = image_names[idx]
        score = similarities[idx]
        row = dataframe[dataframe['image'] == img_name].iloc[0]
        
        results.append({
            'image_name': img_name,
            'score': float(score),
            'display_name': row['display name'],
            'description': row['description'],
            'category': row['category']
        })
    
    return results


def print_results(query, results):
    """вывод результатов в консоль"""
    print("\n" + "="*80)
    print(f"РЕЗУЛЬТАТЫ ПОИСКА: '{query}'")
    print("="*80 + "\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}.  {result['display_name']}")
        print(f"    Категория: {result['category']}")
        print(f"    {result['description'][:70]}...")
        print(f"    Score: {result['score']:.4f}")
        print(f"    Файл: {result['image_name']}")
        print()


def main():
    """Основная функция для запуска из командной строки"""
    # Загружаем модель
    print("Загрузка модели...")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Загружаем checkpoint
    checkpoint_path = os.path.join(PATHS['checkpoints'], 'clip_best.pt')
    
    # Если нет best, берем последний epoch
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(PATHS['checkpoints'], 'clip_epoch_4.pt')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Безопасный вывод test_score
        if 'test_score' in checkpoint:
            print(f"✓ Загружен checkpoint (Test Score: {checkpoint['test_score']:.2f})")
        elif 'history' in checkpoint and 'test_score' in checkpoint['history']:
            test_score = checkpoint['history']['test_score'][-1]
            print(f"✓ Загружен checkpoint (Test Score: {test_score:.2f})")
        else:
            epoch = checkpoint.get('epoch', '?')
            print(f"✓ Загружен checkpoint (эпоха {epoch})")
    else:
        print("⚠ Checkpoint не найден, используется базовая модель")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Загружаем embeddings
    print("Загрузка embeddings...")
    embeddings_path = os.path.join(PATHS['processed_data'], 'image_embeddings.npz')
    data = load_embeddings(embeddings_path)
    print(f"Загружено {len(data['image_names'])} изображений\n")
    
    # Интерактивный поиск
    print("="*80)
    print("СИСТЕМА ПОИСКА ТОВАРОВ")
    print("="*80)
    print("Введите текстовый запрос (или 'exit' для выхода)\n")
    
    while True:
        query = input("🔍 Запрос: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("\nПоиск окончен 👋")
            break
        
        if not query:
            continue
        
        # Поиск
        results = search_products(
            query=query,
            model=model,
            processor=processor,
            image_embeddings=data['embeddings'],
            image_names=data['image_names'],
            dataframe=data['dataframe'],
            top_k=5,
            device=device
        )
        
        # Вывод
        print_results(query, results)


if __name__ == "__main__":
    main()