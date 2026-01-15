"""
Скрипт для скачивания датасета Fashion Product Images с Kaggle
"""
import os
import shutil
import kagglehub
import sys
from pathlib import Path

# Добавляем корневую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.paths import PATHS


def download_dataset():
    """Скачивает датасет с Kaggle"""
    print("Скачивание датасета...")
    
    try:
        # Скачиваем датасет
        kaggle_path = kagglehub.dataset_download(
            "nirmalsankalana/fashion-product-text-images-dataset"
        )
        
        # Создаем директории
        os.makedirs(PATHS['extracted_data'], exist_ok=True)
        
        # Копируем данные в проект
        for item in os.listdir(kaggle_path):
            src = os.path.join(kaggle_path, item)
            dst = os.path.join(PATHS['extracted_data'], item)
            
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        
        print(f"Датасет скачан: {PATHS['extracted_data']}")
        return kaggle_path
        
    except Exception as e:
        print(f"Ошибка: {e}")
        raise


if __name__ == "__main__":
    download_dataset()