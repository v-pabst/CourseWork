# ПЕРВИЧНЫЙ АНАЛИЗ ТАБЛИЧНЫХ ДАННЫХ

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Настройки отображения
plt.style.use('default')
sns.set_theme(style="whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)

print("ПЕРВИЧНЫЙ АНАЛИЗ НАБОРА ТАБЛИЧНЫХ ДАННЫХ")

df = pd.read_csv('data.csv')
if 'length' in df.columns:
    df = df.drop('length', axis=1)
    print("Признак 'Length' удалён")

#1. Визуализация распределения признаков
print("1. ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЯ ПРИЗНАКОВ")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Числовые признаки: {numeric_cols}")

fig, axes = plt.subplots(1, len(numeric_cols), figsize=(15, 4))
if len(numeric_cols) == 1:
    axes = [axes]

for i, col in enumerate(numeric_cols):
    df[col].dropna().hist(bins=50, ax=axes[i], alpha=0.7, color='steelblue', edgecolor='black')
    axes[i].set_title(f'Распределение {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Частота')

plt.suptitle('Рисунок 1. Гистограммы распределения числовых признаков', fontsize=12)
plt.tight_layout()
plt.savefig('figure_1_distributions.png', dpi=150)
plt.show()

#2. Визуализация признаков (ящики с усами)
print("2. ВИЗУАЛИЗАЦИЯ ПРИЗНАКОВ (BOXPLOT)")

fig, axes = plt.subplots(1, len(numeric_cols), figsize=(15, 4))
if len(numeric_cols) == 1:
    axes = [axes]

for i, col in enumerate(numeric_cols):
    sns.boxplot(y=df[col], ax=axes[i], color='lightcoral')
    axes[i].set_title(f'Ящик с усами: {col}')
    axes[i].set_ylabel(col)

plt.suptitle('Рисунок 2. Боксплоты числовых признаков для выявления выбросов', fontsize=12)
plt.tight_layout()
plt.savefig('figure_2_boxplots.png', dpi=150)
plt.show()

#3. Анализ на наличие пропусков
print("3. АНАЛИЗ НА НАЛИЧИЕ ПРОПУСКОВ")

missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100
missing_table = pd.DataFrame({
    'Признак': missing_counts.index,
    'Количество пропусков': missing_counts.values,
    'Доля пропусков, %': missing_percent.values
})
print(missing_table.to_string(index=False))

# Визуализация пропусков
plt.figure(figsize=(10, 3))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Рисунок 3. Карта пропусков в данных')
plt.tight_layout()
plt.savefig('figure_3_missing_heatmap.png', dpi=150)
plt.show()

#4. Корреляционный анализ
print("4. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")

# Корреляционная матрица
corr_matrix = df[numeric_cols].corr()
print("Корреляционная матрица (Пирсон):")
print(corr_matrix.round(3))

# Тепловая карта
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5, square=True)
plt.title('Рисунок 4. Корреляционная матрица числовых признаков')
plt.tight_layout()
plt.savefig('figure_4_correlation.png', dpi=150)
plt.show()

#5. Устранение дубликатов
print("5. УСТРАНЕНИЕ ДУБЛИКАТОВ")

duplicates_count = df.duplicated().sum()
print(f"Количество полных дубликатов строк: {duplicates_count}")

if duplicates_count > 0:
    df = df.drop_duplicates()
    print(f"Дубликаты удалены. Новый размер: {df.shape}")
else:
    print("Полных дубликатов не обнаружено.")

# Проверка дубликатов по ключевым полям
if 'Date' in df.columns:
    key_dups = df.duplicated(subset=['Date']).sum()
    print(f"Дубликатов по дате: {key_dups}")

#6. Анализ и обработка выбросов (метод IQR)
print("6. АНАЛИЗ И ОБРАБОТКА ВЫБРОСОВ")

outliers_info = []
for col in numeric_cols:
    col_clean = df[col].dropna()
    if len(col_clean) > 0:
        Q1 = col_clean.quantile(0.25)
        Q3 = col_clean.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = col_clean[(col_clean < lower_bound) | (col_clean > upper_bound)]
        outliers_info.append({
            'Признак': col,
            'Q1': round(Q1, 2),
            'Q3': round(Q3, 2),
            'IQR': round(IQR, 2),
            'Нижняя граница': round(lower_bound, 2),
            'Верхняя граница': round(upper_bound, 2),
            'Кол-во выбросов': len(outliers),
            'Доля выбросов, %': round(len(outliers) / len(col_clean) * 100, 2)
        })

outliers_df = pd.DataFrame(outliers_info)
print(outliers_df.to_string(index=False))

# Визуализация выбросов до обработки
plt.figure(figsize=(12, 4))
for i, col in enumerate(numeric_cols):
    plt.subplot(1, len(numeric_cols), i+1)
    sns.boxplot(y=df[col], color='salmon')
    plt.title(f'{col} (до обработки)')
plt.suptitle('Рисунок 5. Выбросы до обработки', fontsize=12)
plt.tight_layout()
plt.savefig('figure_5_outliers_before.png', dpi=150)
plt.show()

# Обработка выбросов (цензурирование - Winsorization)
df_processed = df.copy()
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Цензурирование: заменяем выбросы на границы
    df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)

print("\nВыбросы обработаны методом цензурирования (Winsorization)")

# Визуализация после обработки
plt.figure(figsize=(12, 4))
for i, col in enumerate(numeric_cols):
    plt.subplot(1, len(numeric_cols), i+1)
    sns.boxplot(y=df_processed[col], color='lightgreen')
    plt.title(f'{col} (после обработки)')
plt.suptitle('Рисунок 6. Выбросы после обработки', fontsize=12)
plt.tight_layout()
plt.savefig('figure_6_outliers_after.png', dpi=150)
plt.show()

# #7. Фильтрация данных
print("7. ФИЛЬТРАЦИЯ ДАННЫХ")

# ФИЛЬТР 1: Только игры с посещаемостью больше 15000 зрителей
print("ФИЛЬТР 1: Attendance > 15000")

# Создаём копию для фильтрации
df_filter1 = df.dropna(subset=['Attendance'])
size_before1 = len(df_filter1)
df_filter1 = df_filter1[df_filter1['Attendance'] > 15000]
size_after1 = len(df_filter1)

print(f"До фильтра: {size_before1} записей")
print(f"После фильтра: {size_after1} записей")
print(f"Удалено: {size_before1 - size_after1} записей")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# График до фильтрации
axes[0].hist(df['Attendance'].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(15000, color='red', linestyle='--', linewidth=2, label='Граница фильтра (15000)')
axes[0].set_title(f'ДО фильтрации (n={size_before1})')
axes[0].set_xlabel('Посещаемость')
axes[0].set_ylabel('Частота')
axes[0].legend()

# График после фильтрации
axes[1].hist(df_filter1['Attendance'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1].set_title(f'ПОСЛЕ фильтрации (n={size_after1})')
axes[1].set_xlabel('Посещаемость')
axes[1].set_ylabel('Частота')

plt.suptitle('Фильтр 1: Attendance > 15000', fontsize=14)
plt.tight_layout()
plt.savefig('figure_filter1_attendance.png', dpi=150)
plt.show()

# ФИЛЬТР 2: Только победы хозяев (HomeGoals > AwayGoals)
print("ФИЛЬТР 2: Победа хозяев (HomeGoals > AwayGoals)")

size_before2 = len(df)
df_filter2 = df[df['HomeGoals'] > df['AwayGoals']]
size_after2 = len(df_filter2)

print(f"До фильтра: {size_before2} записей")
print(f"После фильтра: {size_after2} записей")
print(f"Удалено: {size_before2 - size_after2} записей")

df['goal_diff'] = df['HomeGoals'] - df['AwayGoals']
df_filter2['goal_diff'] = df_filter2['HomeGoals'] - df_filter2['AwayGoals']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# График до фильтрации
axes[0].hist(df['goal_diff'].dropna(), bins=range(-15, 16), color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Граница фильтра (0)')
axes[0].set_title(f'ДО фильтрации (n={size_before2})')
axes[0].set_xlabel('Разница голов (Home - Away)')
axes[0].set_ylabel('Частота')
axes[0].legend()

# График после фильтрации
axes[1].hist(df_filter2['goal_diff'], bins=range(1, 16), color='lightgreen', edgecolor='black', alpha=0.7)
axes[1].set_title(f'ПОСЛЕ фильтрации (n={size_after2})')
axes[1].set_xlabel('Разница голов (Home - Away)')
axes[1].set_ylabel('Частота')

plt.suptitle('Фильтр 2: HomeGoals > AwayGoals (победа хозяев)', fontsize=14)
plt.tight_layout()
plt.savefig('figure_filter2_home_win.png', dpi=150)
plt.show()

# ФИЛЬТР 3: Результативные игры (сумма голов > 7)
print("ФИЛЬТР 3: Результативная игра (HomeGoals + AwayGoals > 7)")

# Признак "сумма голов"
df['total_goals'] = df['HomeGoals'] + df['AwayGoals']

size_before3 = len(df)
df_filter3 = df[df['total_goals'] > 7]
size_after3 = len(df_filter3)

print(f"До фильтра: {size_before3} записей")
print(f"После фильтра: {size_after3} записей")
print(f"Удалено: {size_before3 - size_after3} записей")
print(f"Доля результативных матчей: {size_after3/size_before3*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# График до фильтрации
axes[0].hist(df['total_goals'], bins=range(0, 20), color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(7, color='red', linestyle='--', linewidth=2, label='Граница фильтра (7)')
axes[0].set_title(f'ДО фильтрации (n={size_before3})')
axes[0].set_xlabel('Сумма голов (Home + Away)')
axes[0].set_ylabel('Частота')
axes[0].legend()

# График после фильтрации
axes[1].hist(df_filter3['total_goals'], bins=range(8, 20), color='lightgreen', edgecolor='black', alpha=0.7)
axes[1].set_title(f'ПОСЛЕ фильтрации (n={size_after3})')
axes[1].set_xlabel('Сумма голов (Home + Away)')
axes[1].set_ylabel('Частота')

plt.suptitle('Фильтр 3: HomeGoals + AwayGoals > 7 (результативные матчи)', fontsize=14)
plt.tight_layout()
plt.savefig('figure_filter3_high_scoring.png', dpi=150)
plt.show()

#8. Добавление шума (для оценки устойчивости модели)
print("8. ДОБАВЛЕНИЕ ШУМА")

df_noisy = df.copy()
noise_level = 0.05  # 5% шума

for col in numeric_cols:
    if col in df_noisy.columns:
        std = df_noisy[col].std()
        noise = np.random.normal(0, noise_level * std, size=len(df_noisy))
        df_noisy[f'{col}_noisy'] = df_noisy[col] + noise
        # Для целочисленных признаков (голы) округляем и не допускаем отрицательных
        if col in ['AwayGoals', 'HomeGoals']:
            df_noisy[f'{col}_noisy'] = df_noisy[f'{col}_noisy'].round().clip(lower=0)

print("Добавлен гауссовский шум (5% от стандартного отклонения) к числовым признакам")
print("Новые признаки: AwayGoals_noisy, HomeGoals_noisy, Attendance_noisy")

# Визуализация влияния шума
fig, axes = plt.subplots(1, len(numeric_cols), figsize=(15, 4))
if len(numeric_cols) == 1:
    axes = [axes]

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col].dropna(), bins=40, alpha=0.5, label='Исходные', color='blue')
    axes[i].hist(df_noisy[f'{col}_noisy'].dropna(), bins=40, alpha=0.5, label='С шумом', color='red')
    axes[i].set_title(f'{col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Частота')
    axes[i].legend()

plt.suptitle('Рисунок 8. Влияние добавления гауссовского шума (5%)', fontsize=12)
plt.tight_layout()
plt.savefig('figure_8_noise_effect.png', dpi=150)
plt.show()

# Статистическая оценка шума
print("\nОценка влияния шума:")
for col in numeric_cols:
    original_mean = df[col].mean()
    noisy_mean = df_noisy[f'{col}_noisy'].mean()
    print(f"{col}: исходное среднее = {original_mean:.2f}, со шумом = {noisy_mean:.2f} (изменение {abs(original_mean - noisy_mean):.3f})")