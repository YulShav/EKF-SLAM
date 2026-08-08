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

adapter = TerrasentiaDataAdapter(Path("ts_2023_06_15.bag"))

ekf = EKF()
initial_pose_set = False

estimations = []
ground_truth = []

for action, value, timestamp in adapter:
    if action == 'predict':
        if initial_pose_set:
            ekf.predict(value)
            
    elif action == 'update':
        if not initial_pose_set:
            ekf.set_pose(value[:2])
            initial_pose_set = True
        else:
            ekf.update(value, mask=[True, True, False])
            estimations.append(ekf.x.copy())
            
    elif action == 'ground_truth':
        ground_truth.append(value[:2])

estimations = np.array(estimations)
ground_truth = np.array(ground_truth)

if len(estimations) == 0:
    print("Нет оценок EKF, используется одометрия")
    estimations = []
    for action, value, timestamp in adapter:
        if action == 'update':
            estimations.append(value[:2])
    estimations = np.array(estimations)

if len(ground_truth) > 0 and len(estimations) > 0:
    offset_x = ground_truth[0, 0] - estimations[0, 0]
    offset_y = ground_truth[0, 1] - estimations[0, 1]
    estimations_aligned = estimations.copy()
    estimations_aligned[:, 0] += offset_x
    estimations_aligned[:, 1] += offset_y


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


ax1.plot(ground_truth[:, 0], ground_truth[:, 1], 'g-', linewidth=2, label='Ground Truth')
ax1.plot(estimations_aligned[:, 0], estimations_aligned[:, 1], 'b-', linewidth=2, label='EKF')
ax1.set_xlabel('X (м)')
ax1.set_ylabel('Y (м)')
ax1.set_title('Сравнение траекторий')
ax1.legend()
ax1.grid(True)
ax1.axis('equal')

min_len = min(len(estimations), len(ground_truth))
error = np.sqrt((estimations[:min_len, 0] + offset_x - ground_truth[:min_len, 0])**2 + 
                (estimations[:min_len, 1] + offset_y - ground_truth[:min_len, 1])**2)

ax2.plot(error, 'r-', linewidth=2)
ax2.set_xlabel('Номер измерения')
ax2.set_ylabel('Ошибка (м)')
ax2.set_title(f'Ошибка EKF (средняя: {np.mean(error):.2f} м, макс: {np.max(error):.2f} м)')
ax2.grid(True)

plt.tight_layout()
Path("results").mkdir(exist_ok=True)
plt.savefig(Path("results") / "comparison_1.png", dpi=150, bbox_inches='tight')
plt.show()