import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import importlib

import sys
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

import ekf.class_ekf
importlib.reload(ekf.class_ekf)
from ekf.class_ekf import EKF

import data.citrus_farm_data_adapter
importlib.reload(data.citrus_farm_data_adapter)
from data.citrus_farm_data_adapter import CitrusFarmDataAdapter

bag_path = root_path / "citrus_farm_7_14.bag"
# bag_path = root_path / "citrus_farm_26_48.bag"
adapter = CitrusFarmDataAdapter(bag_path)

ekf = EKF()
initial_pose_set = False

estimations = []
ground_truth = []

# Диагностика данных
counts = {'predict': 0, 'update': 0, 'ground_truth': 0}
first_timestamps = {}
last_timestamps = {}
dt_values = []

for action, value, timestamp in adapter:
    # Сбор диагностики
    counts[action] = counts.get(action, 0) + 1
    
    if action not in first_timestamps:
        first_timestamps[action] = timestamp
    last_timestamps[action] = timestamp
    
    if action == 'predict':
        dt_values.append(abs(value[0]) + abs(value[1]))  # Сумма приращений
    
    # ===== ОСНОВНАЯ ЛОГИКА =====
    if action == 'predict':
        if initial_pose_set:
            ekf.predict(value)

    elif action == 'update':
        if not initial_pose_set:
            ekf.set_pose(value)
            initial_pose_set = True
        else:
            ekf.update(value)
        estimations.append(ekf.x[:2].copy())

    elif action == 'ground_truth':
        ground_truth.append(value[:2].copy())

print("=== ДИАГНОСТИКА ===")
for action, count in counts.items():
    print(f"{action}: {count} сообщений")
    if action in first_timestamps:
        print(f"  Первое: {first_timestamps[action]/1e9:.3f} сек")
        print(f"  Последнее: {last_timestamps[action]/1e9:.3f} сек")

if dt_values:
    print(f"\nСреднее приращение predict: {np.mean(dt_values):.6f}")
    print(f"Максимальное приращение: {np.max(dt_values):.6f}")
    print(f"Минимальное приращение: {np.min(dt_values):.6f}")

estimations = np.array(estimations)
ground_truth = np.array(ground_truth)

print(f"EKF оценок: {len(estimations)}")
print(f"Ground Truth точек: {len(ground_truth)}")

if len(estimations) == 0:
    print("Нет данных EKF!")
    exit()

if len(ground_truth) == 0:
    print("Нет данных Ground Truth!")
    exit()

offset_x = ground_truth[0, 0] - estimations[0, 0]
offset_y = ground_truth[0, 1] - estimations[0, 1]
estimations_aligned = estimations.copy()
estimations_aligned[:, 0] += offset_x
estimations_aligned[:, 1] += offset_y

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'g-', linewidth=2, label='Ground Truth')
plt.plot(estimations_aligned[:, 0], estimations_aligned[:, 1], 'b-', linewidth=2, label='EKF')
plt.xlabel('X (м)')
plt.ylabel('Y (м)')
plt.title('Сравнение траекторий')
plt.legend()
plt.grid(True)
plt.axis('equal')

min_len = min(len(estimations), len(ground_truth))
error = np.sqrt((estimations[:min_len, 0] + offset_x - ground_truth[:min_len, 0])**2 +
                (estimations[:min_len, 1] + offset_y - ground_truth[:min_len, 1])**2)

plt.subplot(1, 2, 2)
plt.plot(error, 'r-', linewidth=2)
plt.xlabel('Номер измерения')
plt.ylabel('Ошибка (м)')
plt.title(f'Ошибка EKF (средняя: {np.mean(error):.2f} м, макс: {np.max(error):.2f} м)')
plt.grid(True)

plt.tight_layout()
# plt.savefig(Path("results") / 'citrus_ekf_result.png', dpi=150)
plt.show()