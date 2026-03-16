# -*- coding: utf-8 -*-
"""
Лабораторная работа №2
Датасет: gaming_laptops_2026_q1.csv
Этапы:
1. Обработка пропусков
2. Кодирование категориальных признаков (OneHotEncoder)
3. Масштабирование числовых признаков
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 1. Загрузка и первичный анализ
df = pd.read_csv('/content/gaming_laptops_2026_q1.csv')

print("Размер исходного датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())

print("\nИнформация о типах данных и пропусках:")
print(df.info())

# -------------------------------
# 2. Анализ пропусков
# -------------------------------
print("\nКоличество пропусков в каждом столбце:")
missing = df.isnull().sum()
print(missing[missing > 0])

missing_pct = (missing[missing > 0] / len(df)) * 100
print("\nДоля пропусков (%) в столбцах с пропусками:")
print(missing_pct)

# -------------------------------
# 3. Обработка пропусков
# -------------------------------
df_clean = df.copy()

# 3.1 Числовые столбцы – заполнение медианой
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
numeric_imputer = SimpleImputer(strategy='median')
df_clean[numeric_cols] = numeric_imputer.fit_transform(df_clean[numeric_cols])

# 3.2 Категориальные столбцы – заполнение модой (самым частым значением)
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
categorical_imputer = SimpleImputer(strategy='most_frequent')
df_clean[categorical_cols] = categorical_imputer.fit_transform(df_clean[categorical_cols])

print("\nПропуски после обработки (сумма):", df_clean.isnull().sum().sum())

# 4. Кодирование категориальных признаков (OneHotEncoder)

# Создаём dummy  переменные для brand
onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
brand_onehot = onehot.fit_transform(df_clean[['brand']])

# Преобразуем в DataFrame с понятными названиями столбцов
brand_onehot_df = pd.DataFrame(
    brand_onehot,
    columns=onehot.get_feature_names_out(['brand']),
    index=df_clean.index
)

# Объединяем с основным датафреймом и удаляем исходный столбец brand
df_encoded = pd.concat([df_clean, brand_onehot_df], axis=1)
df_encoded.drop('brand', axis=1, inplace=True)

print("\nРазмер после OneHot-кодирования brand:", df_encoded.shape)
print("Новые столбцы с brand:")
print(brand_onehot_df.columns.tolist())


# 5. Масштабирование числовых признаков
# Приблизительно нормальное распределение
# используем z-нормализация, оно преобразует данные так, чтобы среднее арифметическое стало равно 0, а стандартное отклонение — 1
# Масштабируем только исходные числовые признаки (numeric_cols).
scaler = StandardScaler()
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

# Проверим результаты масштабирования (среднее ≈ 0, ст. отклонение ≈ 1)
print("\nСредние значения после масштабирования (первые 5 числовых признаков):")
print(df_encoded[numeric_cols].mean().round(2).head())
print("\nСтандартные отклонения после масштабирования (первые 5 числовых признаков):")
print(df_encoded[numeric_cols].std().round(2).head())

# -------------------------------
# 6. Итоговый результат
# -------------------------------
print("\nИтоговый размер обработанного датасета:", df_encoded.shape)
print("\nПервые 5 строк обработанного датасета:")
print(df_encoded.head())

# При необходимости можно сохранить результат в CSV
# df_encoded.to_csv('gaming_laptops_processed.csv', index=False)