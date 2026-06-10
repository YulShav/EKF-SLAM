import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import importlib
from scipy.spatial import KDTree
import ekf.class_ekf
importlib.reload(ekf.class_ekf)
from ekf.class_ekf import EKF
import data.ford_data_adapter
importlib.reload(data.ford_data_adapter)
from data.ford_data_adapter import FordDataAdapter
import sys
from pathlib import Path


adapter = FordDataAdapter(Path("2017-10-26-V2-Log6-sorted.bag"))

ekf_raw = EKF()
raw_x, raw_y = [], []
raw_pose_set = False

initial_pose_set = False

ekf = EKF()

estimations_x = []
estimations_y = []
ground_truth_x = []
ground_truth_y = []

for i, (action, value, timestamp) in enumerate(adapter):
    if action == 'predict':
        if initial_pose_set:
            ekf.predict(value)
        if raw_pose_set:
            ekf_raw.predict(value)
            raw_x.append(ekf_raw.x[0])
            raw_y.append(ekf_raw.x[1])

    elif action == 'update':
        if not initial_pose_set:
            ekf.set_pose(value)
            initial_pose_set = True
        else:
            ekf.update(value)
        estimations_x.append(ekf.x[0])
        estimations_y.append(ekf.x[1])

        if not raw_pose_set:
            ekf_raw.set_pose(value)
            raw_pose_set = True
        else:
            ekf_raw.update(value)
            raw_x.append(ekf_raw.x[0])
            raw_y.append(ekf_raw.x[1])

    elif action == 'ground_truth':
        ground_truth_x.append(value[0])
        ground_truth_y.append(value[1])

    if i > 30000:
        break

def compute_errors(ref_x, ref_y, est_x, est_y):
    ref = np.array(list(zip(ref_x, ref_y)))
    est = np.array(list(zip(est_x, est_y)))
    if len(ref) == 0 or len(est) == 0:
        return None, None
    tree = KDTree(ref)
    distances, _ = tree.query(est)
    metrics = {
        'mean': np.mean(distances),
        'percentile_95': np.percentile(distances, 95),
        'max': np.max(distances),
    }
    return distances, metrics

dist_raw, metrics_raw = compute_errors(ground_truth_x, ground_truth_y, raw_x, raw_y)
dist_ekf, metrics_ekf = compute_errors(ground_truth_x, ground_truth_y, estimations_x, estimations_y)
dist_compare, metrics_compare = compute_errors(raw_x, raw_y, estimations_x, estimations_y)

Path("results").mkdir(exist_ok=True)
with open(Path("results") / "metrics.txt", 'w', encoding='utf-8') as f:
    original_stdout = sys.stdout
    sys.stdout = f
    
    print("Сравнение raw_odometry и ground_truth")
    if metrics_raw:
        print(f"   Среднее расстояние: {metrics_raw['mean']:.3f} м")
        print(f"   95-й персентиль:    {metrics_raw['percentile_95']:.3f} м")
        print(f"   Максимум:           {metrics_raw['max']:.3f} м")

    print("\nСравнение ekf и ground_truth")
    if metrics_ekf:
        print(f"   Среднее расстояние: {metrics_ekf['mean']:.3f} м")
        print(f"   95-й персентиль:    {metrics_ekf['percentile_95']:.3f} м")
        print(f"   Максимум:           {metrics_ekf['max']:.3f} м")

    print("\nRAW и EKF (насколько изменилась траектория)")
    if metrics_compare:
        print(f"   Среднее расхождение: {metrics_compare['mean']:.6f} м")
        print(f"   95-й персентиль:     {metrics_compare['percentile_95']:.6f} м")

    if metrics_raw and metrics_ekf:
        improvement = (1 - metrics_ekf['mean'] / metrics_raw['mean']) * 100
        print(f"\nУЛ: {improvement:+.1f}%")
        if improvement > 0:
            print("EKF улучшил точность")
        else:
            print("EKF ухудшил точность")

    f.write("\nRaw vs GT:\n")
    f.write(f"  mean: {metrics_raw['mean']:.4f}, 95%: {metrics_raw['percentile_95']:.4f}, max: {metrics_raw['max']:.4f}\n")
    f.write("EKF vs GT:\n")
    f.write(f"  mean: {metrics_ekf['mean']:.4f}, 95%: {metrics_ekf['percentile_95']:.4f}, max: {metrics_ekf['max']:.4f}\n")
    f.write(f"Improvement: {improvement:.1f}%\n")

    sys.stdout = original_stdout

plt.figure(figsize=(10, 8))
plt.plot(ground_truth_x, ground_truth_y, 'g-', linewidth=5, label='Ground Truth')
plt.plot(estimations_x, estimations_y, 'b-', linewidth=2, label='EKF')
plt.xlabel('X (м)')
plt.ylabel('Y (м)')
plt.title('Сравнение EKF с эталонной траекторией')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.savefig(Path("results") / "comparison_2017-10-26-V2-Log6.png", dpi=150, bbox_inches='tight')
plt.show()
plt.close()