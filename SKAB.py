import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

#1. Загрузка и первичное знакомство
file_path = 'data/valve1/0.csv'
df = pd.read_csv(file_path, sep=';', parse_dates=['datetime'])
df.set_index('datetime', inplace=True)

print("Первые 5 строк данных:")
print(df.head(5))

print(f"\nРазмер: {df.shape[0]:,} строк, {df.shape[1]} столбцов")

print("\nСтолбцы данных:")
for i, col in enumerate(df.columns):
    print(f"   {i+1}. {col}")

n_anomaly = df['anomaly'].sum()
print("\nБаланс классов:")
print(f"Всего наблюдений: {len(df)}")
print(f"Норма: {len(df) - n_anomaly} ({(len(df)-n_anomaly)/len(df)*100:.2f}%)")
print(f"Аномалии: {n_anomaly} ({n_anomaly/len(df)*100:.2f}%)")

time_diffs = df.index.to_series().diff().dt.total_seconds()
print("\nЧастота дискретизации")
print(f"Интервал: {time_diffs.mode()[0]:.0f} секунд")

#2. Визуализация исходных данных
#Список сенсорных каналов
sensors = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 
           'Pressure', 'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS']


#Многопанельный график 
channels = ['Temperature', 'Pressure', 'Current']
titles = ['Температура двигателя (°C)', 'Давление в контуре (Bar)', 'Ток двигателя (A)']
colors = ['red', 'blue', 'green']

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

for i, (channel, title, color) in enumerate(zip(channels, titles, colors)):
    axes[i].plot(df.index, df[channel], color=color, linewidth=1)
    axes[i].fill_between(df.index, df[channel].min(), df[channel].max(),
                         where=df['anomaly'] == 1, alpha=0.3, color='red')
    axes[i].set_ylabel(title, fontsize=10)
    axes[i].grid(True, alpha=0.3)
    # Ключевой момент: каждый подграфик имеет свой независимый масштаб
    axes[i].autoscale(enable=True, axis='y')

axes[-1].set_xlabel('Дата и время', fontsize=12)
plt.suptitle('Многомерный временной ряд SKAB', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#3. Статистический анализ 
#Статистические характеристики каналов

print("\n1. СТАТИСТИЧЕСКИЕ ХАРАКТЕРИСТИКИ КАНАЛОВ:")

stats_data = []
for sensor in sensors:
    mean_val = df[sensor].mean()
    std_val = df[sensor].std()
    min_val = df[sensor].min()
    q1_val = df[sensor].quantile(0.25)
    q2_val = df[sensor].median()
    q3_val = df[sensor].quantile(0.75)
    max_val = df[sensor].max()
    count_val = df[sensor].count()
    
    stats_data.append({
        'Канал': sensor,
        'count': count_val,
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'Q1': q1_val,
        'Q2 (median)': q2_val,
        'Q3': q3_val,
        'max': max_val
    })
stats_df = pd.DataFrame(stats_data)
stats_df = stats_df.set_index('Канал')

#Вывод таблицы с округлением до 4 знаков
print(stats_df.round(4).to_string())

#Частота дискретизации

print("\n2. ЧАСТОТА ДИСКРЕТИЗАЦИИ:")
time_diffs = df.index.to_series().diff().dt.total_seconds()

#Частота дискретизации 
sampling_freq = 1 / time_diffs.median()

print(f"   Частота дискретизации: {sampling_freq:.2f} Гц")
print(f"   Средний интервал: {time_diffs.mean():.4f} секунд")
print(f"   Медианный интервал: {time_diffs.median():.4f} секунд")
print(f"   Стандартное отклонение интервалов: {time_diffs.std():.6f} секунд")
print(f"   Интервалы равномерны: {'Да' if time_diffs.std() < 0.01 else 'Нет'}")

# Проверка на пропуски во времени
expected_index = pd.date_range(start=df.index[0], end=df.index[-1], freq='S')
missing_times = expected_index.difference(df.index)
if len(missing_times) > 0:
    print(f"   ВНИМАНИЕ: Обнаружено {len(missing_times)} пропущенных временных меток")
else:
    print(f"   Пропущенных временных меток нет: ряд полный и непрерывный")

#4. Анализ пропусков и выбросов 
#пропуски
print("\n1. пропуски в данных:")
for sensor in sensors:
    missing = df[sensor].isnull().sum()
    print(f"   - {sensor:25}: {missing} пропусков ({missing/len(df)*100:.2f}%)")

#выбросы по правилу 3 сигм
print("\n2. выбросы по правилу 3σ:")
for sensor in sensors:
    mean = df[sensor].mean()
    std = df[sensor].std()
    outliers = df[(df[sensor] < mean - 3*std) | (df[sensor] > mean + 3*std)]
    pct = len(outliers) / len(df) * 100
    print(f"   - {sensor:25}: {len(outliers):4d} выбросов ({pct:.2f}%)")

# boxplot для всех каналов
plt.figure(figsize=(14, 6))
df[sensors].boxplot(rot=45)
plt.title('Диаграммы размаха для всех сенсорных каналов')
plt.ylabel('Значение')
plt.grid(True, alpha=0.3) 
plt.tight_layout() 
plt.show()

# #5. Анализ диапазонов значений 
#Диапазоны значений
print("\n1. ДИАПАЗОНЫ ЗНАЧЕНИЙ:")
for sensor in sensors:
    min_val = df[sensor].min()
    max_val = df[sensor].max()
    range_val = max_val - min_val
    print(f"   - {sensor:25}: [{min_val:8.4f}, {max_val:8.4f}], размах = {range_val:8.4f}")

#Сравнение масштабов
print("\n2. СРАВНЕНИЕ МАСШТАБОВ (относительно Temperature):")
temp_range = df['Temperature'].max() - df['Temperature'].min()
for sensor in sensors:
    s_range = df[sensor].max() - df[sensor].min()
    ratio = temp_range / s_range if s_range > 0 else float('inf')
    print(f"   - {sensor:25}: размах в {ratio:8.1f} раз {'меньше' if ratio > 1 else 'больше'} чем у Temperature")

#Визуальное сравнение (нормализованный boxplot)
fig, ax = plt.subplots(figsize=(14, 5))
df_norm = (df[sensors] - df[sensors].min()) / (df[sensors].max() - df[sensors].min())
df_norm.boxplot(ax=ax, rot=45)
ax.set_title('Диапазоны каналов после нормализации к [0,1]')
ax.set_ylabel('Нормализованное значение')
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.show()

#6. Корреляционный анализ 
#Матрица корреляции
corr_matrix = df[sensors].corr()
print("\n1. МАТРИЦА КОРРЕЛЯЦИИ ПИРСОНА:")
print(corr_matrix.round(3))

#Тепловая карта
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5)
plt.title('Матрица корреляции между сенсорными каналами', fontsize=14)
plt.tight_layout()
plt.show()

#Сильные корреляции
print("\n2. СИЛЬНЫЕ КОРРЕЛЯЦИИ (|r| > 0.8):")
strong = []
for i, sens1 in enumerate(sensors):
    for j, sens2 in enumerate(sensors):
        if i < j:
            r = corr_matrix.loc[sens1, sens2]
            if abs(r) > 0.8:
                strong.append((sens1, sens2, r))
                print(f"   - {sens1} ↔ {sens2}: r = {r:.4f}")

#Слабые корреляции
print("\n3. СЛАБЫЕ КОРРЕЛЯЦИИ (|r| < 0.2):")
for i, sens1 in enumerate(sensors):
    for j, sens2 in enumerate(sensors):
        if i < j:
            r = corr_matrix.loc[sens1, sens2]
            if abs(r) < 0.2:
                print(f"   - {sens1} ↔ {sens2}: r = {r:.4f}")

# #7. Поиск и анализ шумов 
ts = df['Temperature'].copy()
ts = ts.asfreq('S')
ts = ts.interpolate(method='linear')

#Выполнение декомпозиции
decomp = seasonal_decompose(ts, model='additive', period=60)
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

#Исходный ряд
axes[0].plot(ts, color='black', linewidth=0.8)
axes[0].set_ylabel('Исходный ряд', fontsize=10)
axes[0].set_title('Исходный ряд (температура)', fontsize=11)
axes[0].grid(True, alpha=0.3)

#Тренд
axes[1].plot(decomp.trend, color='green', linewidth=1)
axes[1].set_ylabel('Тренд', fontsize=10)
axes[1].set_title('Трендовая компонента', fontsize=11)
axes[1].grid(True, alpha=0.3)

#Сезонность
axes[2].plot(decomp.seasonal, color='orange', linewidth=0.8)
axes[2].set_ylabel('Сезонность', fontsize=10)
axes[2].set_title('Сезонная компонента (период 60 сек)', fontsize=11)
axes[2].grid(True, alpha=0.3)

#Шум
axes[3].plot(decomp.resid, color='red', linewidth=0.5)
axes[3].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
axes[3].set_ylabel('Шум', fontsize=10)
axes[3].set_title('Остатки (шумовая компонента)', fontsize=11)
axes[3].set_xlabel('Дата и время', fontsize=10)
axes[3].grid(True, alpha=0.3)

plt.suptitle('Декомпозиция временного ряда температуры', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n4. РАСЧЁТ ОТНОШЕНИЯ СИГНАЛ/ШУМ (SNR):")
signal = decomp.trend + decomp.seasonal
noise = decomp.resid
signal_clean = signal.dropna()
noise_clean = noise.dropna()
var_signal = np.var(signal_clean)
var_noise = np.var(noise_clean)
SNR = 10 * np.log10(var_signal / var_noise)
print(f"Дисперсия сигнала: {var_signal:.4f}")
print(f"Дисперсия шума: {var_noise:.4f}")
print(f"NR = {SNR:.2f} дБ")

# Интерпретация SNR
if SNR > 20:
    qual = "Отлично (>20 дБ) — шум практически незаметен"
elif SNR > 10:
    qual = "Хорошо (10-20 дБ) — шум присутствует, но сигнал доминирует"
elif SNR > 0:
    qual = "Удовлетворительно (0-10 дБ) — сигнал и шум сравнимы"
else:
    qual = "Плохо (<0 дБ) — шум сильнее сигнала"
print(f"Оценка: {qual}")

#Анализ распределения шума
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Гистограмма
axes[0].hist(noise_clean, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
axes[0].set_title('Распределение шума (гистограмма)', fontsize=12)
axes[0].set_xlabel('Значение шума')
axes[0].set_ylabel('Плотность')
axes[0].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(noise_clean, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q plot (проверка нормальности)', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Статистика шума
skewness = stats.skew(noise_clean.dropna())
kurtosis = stats.kurtosis(noise_clean.dropna())

print(f"\nАсимметрия (skewness): {skewness:.4f}")
print(f"Эксцесс (kurtosis): {kurtosis:.4f}")

# Определение формы распределения
if abs(skewness) < 0.5 and abs(kurtosis) < 3:
    shape = "Нормальное"
    interpretation = "Шум случаен, некоррелирован, фильтрация не требуется"
elif abs(kurtosis) > 3:
    shape = "С тяжёлыми хвостами"
    interpretation = "Есть редкие выбросы, для фильтрации использовать медианный фильтр"
elif abs(skewness) > 1:
    shape = "Асимметричное"
    interpretation = "Требуется преобразование (логарифмирование)"
else:
    shape = "Условно нормальное"
    interpretation = "Можно работать с исходными данными"

print(f"\nФорма распределения: {shape}")
print(f"Интерпретация: {interpretation}")
