import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import importlib
from scipy.interpolate import interp1d

import sys
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

import ekf.class_ekf
importlib.reload(ekf.class_ekf)
from ekf.class_ekf import EKF

import data.citrus_farm_data_adapter
importlib.reload(data.citrus_farm_data_adapter)
from data.citrus_farm_data_adapter import CitrusFarmDataAdapter

USE_EVERY_N = 40

bag_path = root_path / "citrus_farm_7_14.bag"
adapter = CitrusFarmDataAdapter(bag_path, use_every_n_rtk=USE_EVERY_N)

ekf = EKF()
initial_pose_set = False

est_times = []
est_x = []
est_y = []

gt_times = []
gt_x = []
gt_y = []

counts = {'predict': 0, 'update': 0, 'ground_truth': 0}

for action, value, timestamp in adapter:
    counts[action] = counts.get(action, 0) + 1
    
    if action == 'predict':
        if initial_pose_set:
            ekf.predict(value)

    elif action == 'update':
        if not initial_pose_set:
            ekf.set_pose(value)
            initial_pose_set = True
        else:
            ekf.update(value)

        est_times.append(timestamp)
        est_x.append(ekf.x[0])
        est_y.append(ekf.x[1])

    elif action == 'ground_truth':
        gt_times.append(timestamp)
        gt_x.append(value[0])
        gt_y.append(value[1])

print(f"Обработано: Predict={counts['predict']}, Update={counts['update']}, GT={counts['ground_truth']}")
print(f"RTK использовано для update: {counts['update']} раз")
print(f"RTK использовано для ground_truth: {counts['ground_truth']} раз")

est_times = np.array(est_times)
est_x = np.array(est_x)
est_y = np.array(est_y)

gt_times = np.array(gt_times)
gt_x = np.array(gt_x)
gt_y = np.array(gt_y)

offset_x = gt_x[0] - est_x[0]
offset_y = gt_y[0] - est_y[0]

est_x_aligned = est_x + offset_x
est_y_aligned = est_y + offset_y

interp_x = interp1d(est_times, est_x_aligned, kind='linear', fill_value="extrapolate")
interp_y = interp1d(est_times, est_y_aligned, kind='linear', fill_value="extrapolate")

ekf_at_gt_x = interp_x(gt_times)
ekf_at_gt_y = interp_y(gt_times)

# 3. Расчет ошибки
error = np.sqrt((ekf_at_gt_x - gt_x)**2 + (ekf_at_gt_y - gt_y)**2)


print(f"Средняя ошибка:  {np.mean(error):.3f} м")
print(f"Медианная ошибка:{np.median(error):.3f} м")
print(f"Максимальная ошибка: {np.max(error):.3f} м")
print(f"Стандартное отклонение: {np.std(error):.3f} м")
print("="*60)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(gt_x, gt_y, 'g-', linewidth=3, label='Ground Truth')
plt.plot(est_x_aligned, est_y_aligned, 'b-', linewidth=2, label='EKF')

plt.xlabel('X (м)')
plt.ylabel('Y (м)')
plt.title(f'Сравнение траекторий')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.plot(gt_times, error, 'r-', linewidth=2)
plt.xlabel('Время (нс)')
plt.ylabel('Ошибка (м)')
plt.title(f'Ошибка EKF во времени\n(средняя: {np.mean(error):.3f} м, макс: {np.max(error):.3f} м)')
plt.grid(True)

plt.tight_layout()
plt.savefig(Path("results") / f'citrus_ekf_res.png', dpi=150)
plt.show()