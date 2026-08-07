import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import importlib

import ekf.class_ekf
importlib.reload(ekf.class_ekf)
from ekf.class_ekf import EKF

import data.terrasentia_data_adapter
importlib.reload(data.terrasentia_data_adapter)
from data.terrasentia_data_adapter import TerrasentiaDataAdapter

# Загрузка данных
adapter = TerrasentiaDataAdapter(Path("ts_2023_06_15.bag"))

# Инициализация EKF
ekf = EKF()
initial_pose_set = False

# Массивы
estimations = []
ground_truth = []
updates = []  # Для отладки

# Обработка (как в старом коде)
for action, value, timestamp in adapter:
    if action == 'predict':
        if initial_pose_set:
            ekf.predict(value)
            
    elif action == 'update':
        if not initial_pose_set:
            ekf.set_pose(value)
            initial_pose_set = True
        else:
            ekf.update(value)
        
        estimations.append([ekf.x[0], ekf.x[1]])
        updates.append(value[:2])
        
    elif action == 'ground_truth':
        ground_truth.append(value[:2])

# Конвертация
estimations = np.array(estimations)
ground_truth = np.array(ground_truth)
updates = np.array(updates)

# Если нет EKF, используем update
if len(estimations) == 0:
    print("Используем одометрию")
    estimations = updates

# Выравнивание по первой точке ground truth
if len(ground_truth) > 0 and len(estimations) > 0:
    offset_x = ground_truth[0, 0] - estimations[0, 0]
    offset_y = ground_truth[0, 1] - estimations[0, 1]
    estimations_aligned = estimations.copy()
    estimations_aligned[:, 0] += offset_x
    estimations_aligned[:, 1] += offset_y

# Построение графика
plt.figure(figsize=(12, 5))

# Траектории
plt.subplot(1, 2, 1)
plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'g-', linewidth=2, label='Ground Truth')
plt.plot(estimations_aligned[:, 0], estimations_aligned[:, 1], 'b-', linewidth=2, label='EKF')
plt.xlabel('X (м)')
plt.ylabel('Y (м)')
plt.title('Сравнение траекторий')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Ошибка
if len(ground_truth) > 0 and len(estimations) > 0:
    min_len = min(len(estimations), len(ground_truth))
    error = np.sqrt((estimations[:min_len, 0] + offset_x - ground_truth[:min_len, 0])**2 + 
                    (estimations[:min_len, 1] + offset_y - ground_truth[:min_len, 1])**2)
    
    plt.subplot(1, 2, 2)
    plt.plot(error, 'r-', linewidth=2)
    plt.xlabel('Номер измерения')
    plt.ylabel('Ошибка (м)')
    plt.title(f'Ошибка EKF\nСредняя: {np.mean(error):.2f} м, Макс: {np.max(error):.2f} м')
    plt.grid(True)

plt.tight_layout()
plt.savefig('ekf_result.png', dpi=150)
plt.show()