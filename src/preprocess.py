"""
Модуль для предобработки данных Clothes Dataset
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from config.paths import PATHS


def clean_data(df):
    """Очистка данных от дубликатов и плохих описаний"""
    print(f"Исходно записей: {len(df)}")
    
    # Удаляем плохие описания
    mask = (
        (df['description'].isna()) |
        (df['description'].str.strip() == '') |
        (df['description'].str.strip() == '-') |
        (df['description'].str.lower() == 'style note')
    )
    df = df[~mask]
    print(f"После удаления плохих описаний: {len(df)}")
    
    # Удаляем дубликаты
    df = df.drop_duplicates(subset=['description', 'display name'])
    print(f"После удаления дубликатов: {len(df)}")
    
    
    
    # Сбрасываем индексы
    df = df.reset_index(drop=True)
    
    return df


def split_data(df, test_size=0.1, random_state=42):
    """Разделение на train/test"""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    print(f"\nTrain: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, test_df


def save_split_data(train_df, test_df):
    """Сохранение train/test данных"""
    split_dir = os.path.join(PATHS['processed_data'], 'split')
    os.makedirs(split_dir, exist_ok=True)
    
    train_path = os.path.join(split_dir, 'train.csv')
    test_path = os.path.join(split_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train сохранен: {train_path}")
    print(f"Test сохранен: {test_path}")


def preprocess():
    """Основная функция предобработки"""
    # Загружаем данные
    input_path = os.path.join(PATHS['extracted_data'], 'data.csv')
    df = pd.read_csv(input_path)
    
    # Очищаем
    df = clean_data(df)
    
    # Разделяем на train/test
    train_df, test_df = split_data(df)
    # Добавляем колонку с объединенными текстами
    
    
    train_df['full_text'] = train_df['display name'] + '. ' + train_df['description']
    test_df['full_text'] = test_df['display name'] + '. ' + test_df['description']
    
    # Сохраняем
    save_split_data(train_df, test_df)
    
    return train_df, test_df


if __name__ == "__main__":
    preprocess()