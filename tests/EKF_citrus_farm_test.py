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
N_LIST = [1, 5, 10, 40]

bag_path = root_path / "citrus_farm_7_14.bag"
adapter = CitrusFarmDataAdapter(bag_path, use_every_n_rtk=1)

counts = {'predict': 0, 'update': 0, 'ground_truth': 0}

# for action, value, timestamp in adapter:
#     counts[action] = counts.get(action, 0) + 1
    
#     # if action == 'predict':
#     #     if initial_pose_set:
#     #         ekf.predict(value)
#     if action == 'predict':
#         if initial_pose_set:
#             ekf.predict(value)
#             est_times.append(timestamp)
#             est_x.append(ekf.x[0]); est_y.append(ekf.x[1])

#     elif action == 'update':
#         if not initial_pose_set:
#             ekf.set_pose(value)
#             initial_pose_set = True
#         else:
#             ekf.update(value)

#         est_times.append(timestamp)
#         est_x.append(ekf.x[0])
#         est_y.append(ekf.x[1])

#     elif action == 'ground_truth':
#         gt_times.append(timestamp)
#         gt_x.append(value[0])
#         gt_y.append(value[1])

events = []
for action, value, timestamp in adapter:
    counts[action] = counts.get(action, 0) + 1
    events.append((action, value, timestamp))

print(f"Обработано: Predict={counts['predict']}, Update={counts['update']}, GT={counts['ground_truth']}")

gt_raw = np.array([(ts, v[0], v[1]) for a, v, ts in events if a == 'ground_truth'], dtype=float)
gt_times, gt_idx = np.unique(gt_raw[:, 0], return_index=True)
gt_x = gt_raw[gt_idx, 1]
gt_y = gt_raw[gt_idx, 2]

print(f"Эталонных точек RTK: {len(gt_times)}")


def run_ekf(events, n):
    ekf = EKF()
    initial_pose_set = False

    est_times = []
    est_x = []
    est_y = []
    upd_xy = []

    rtk_counter = 0
    rtk_err = []
    rtk_times = []
    log_P = []
    log_P_after = []
    log_K = []

    for action, value, timestamp in events:
        if action == 'init':
            ekf.set_pose(value)
            initial_pose_set = True
            est_times.append(timestamp); est_x.append(ekf.x[0]); est_y.append(ekf.x[1])

        elif action == 'predict':
            if initial_pose_set:
                ekf.predict(value)
                est_times.append(timestamp); est_x.append(ekf.x[0]); est_y.append(ekf.x[1])

        elif action == 'update':
            if initial_pose_set:
                rtk_counter += 1
                if n is not None and rtk_counter % n == 0:
                    P_before = ekf.P.copy()
                    x_before = ekf.x.copy()
                    ekf.update_pos(value[:2])
                    innovation = value[:2] - x_before[:2]
                    K_gain = (ekf.x[:2] - x_before[:2]) / (innovation + 1e-12)
                    rtk_err.append(np.linalg.norm(ekf.x[:2] - value[:2]))
                    rtk_times.append(timestamp)
                    log_P.append(np.diag(P_before))
                    log_P_after.append(np.diag(ekf.P))
                    log_K.append(K_gain)
                    upd_xy.append([ekf.x[0], ekf.x[1]])
                    est_times.append(timestamp); est_x.append(ekf.x[0]); est_y.append(ekf.x[1])

    return (np.array(est_times), np.array(est_x), np.array(est_y),
            np.array(upd_xy).reshape(-1, 2),
            np.array(rtk_err), np.array(rtk_times),
            np.array(log_P), np.array(log_P_after), np.array(log_K))


def evaluate(est_times, est_x, est_y):
    t, idx = np.unique(est_times, return_index=True)
    ex = est_x[idx]
    ey = est_y[idx]

    mask = (gt_times >= t[0]) & (gt_times <= t[-1])

    interp_x = interp1d(t, ex, kind='linear', fill_value="extrapolate")
    interp_y = interp1d(t, ey, kind='linear', fill_value="extrapolate")

    ekf_at_gt_x = interp_x(gt_times[mask])
    ekf_at_gt_y = interp_y(gt_times[mask])

    error = np.sqrt((ekf_at_gt_x - gt_x[mask])**2 + (ekf_at_gt_y - gt_y[mask])**2)
    return gt_times[mask], error


def stats(error):
    return {
        'mean': np.mean(error),
        'median': np.median(error),
        'p95': np.percentile(error, 95),
        'max': np.max(error),
        'std': np.std(error),
        'rmse': np.sqrt(np.mean(error**2)),
    }


results = {}

est_times, est_x, est_y, upd_xy, rtk_err, rtk_times, log_P, log_P_after, log_K = run_ekf(events, None)
err_t, error = evaluate(est_times, est_x, est_y)
results['Без RTK'] = {'n': None, 'est': (est_times, est_x, est_y), 'upd': upd_xy,
                      'err_t': err_t, 'error': error, 'stats': stats(error),
                      'log_P': log_P, 'log_P_after': log_P_after, 'log_K': log_K,
                      'rtk_err': rtk_err}

for n in N_LIST:
    est_times, est_x, est_y, upd_xy, rtk_err, rtk_times, log_P, log_P_after, log_K = run_ekf(events, n)
    err_t, error = evaluate(est_times, est_x, est_y)
    results[f'N={n}'] = {'n': n, 'est': (est_times, est_x, est_y), 'upd': upd_xy,
                         'err_t': err_t, 'error': error, 'stats': stats(error),
                         'log_P': log_P, 'log_P_after': log_P_after, 'log_K': log_K,
                         'rtk_err': rtk_err}

t0 = gt_times[0]
main_key = f'N={USE_EVERY_N}'
main = results[main_key]
base = results['Без RTK']

header = f"{'Режим':<10}{'Update':>8}{'Период, с':>11}{'Средняя':>10}{'Медиана':>10}{'P95':>10}{'Макс':>10}{'Std':>10}{'RMSE':>10}"
print("\n" + "=" * len(header))
print("Ошибка EKF относительно RTK (все сообщения), м")
print("=" * len(header))
print(header)
print("-" * len(header))

rtk_rate = len(gt_times) / ((gt_times[-1] - gt_times[0]) / 1e9)

for key, r in results.items():
    s = r['stats']
    period = f"{r['n'] / rtk_rate:.2f}" if r['n'] is not None else "-"
    print(f"{key:<10}{len(r['upd']):>8}{period:>11}{s['mean']:>10.3f}{s['median']:>10.3f}"
          f"{s['p95']:>10.3f}{s['max']:>10.3f}{s['std']:>10.3f}{s['rmse']:>10.3f}")

print("=" * len(header))

gain = (base['stats']['mean'] - main['stats']['mean']) / base['stats']['mean'] * 100
print(f"\nОсновной режим: {main_key}")
print(f"Средняя ошибка без RTK: {base['stats']['mean']:.3f} м, с RTK: {main['stats']['mean']:.3f} м")
print(f"Улучшение за счёт RTK: {gain:.1f} %")
print("=" * 60)

log_P = main['log_P']
log_P_after = main['log_P_after']
log_K = main['log_K']

print("Ковариация P (до update), диагональ:")
print(f"  P_xx: mean={log_P[:,0].mean():.4f}, max={log_P[:,0].max():.4f}")
print(f"  P_yy: mean={log_P[:,1].mean():.4f}, max={log_P[:,1].max():.4f}")
print(f"  P_tt: mean={log_P[:,2].mean():.4f}, max={log_P[:,2].max():.4f}")

print("Ковариация P (после update), диагональ:")
print(f"  P_xx: mean={log_P_after[:,0].mean():.4f}")
print(f"  P_yy: mean={log_P_after[:,1].mean():.4f}")

print("Коэффициент усиления K (доля смещения к измерению):")
print(f"  K_x: mean={log_K[:,0].mean():.3f}, max={log_K[:,0].max():.3f}")
print(f"  K_y: mean={log_K[:,1].mean():.3f}, max={log_K[:,1].max():.3f}")

print("Ошибка EKF относительно update-измерения (rtk_err):")
print(f"  mean={main['rtk_err'].mean():.4f}, median={np.median(main['rtk_err']):.4f}, "
      f"p95={np.percentile(main['rtk_err'], 95):.4f}, max={main['rtk_err'].max():.4f}")

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(ax, x, y, cov, n_std=2.0, **kwargs):
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(x, y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


colors = {'Без RTK': 'gray', 'N=1': 'tab:green', 'N=5': 'tab:orange', 'N=10': 'tab:red', 'N=40': 'tab:blue'}
colors = {k: colors.get(k, 'tab:purple') for k in results}

plt.figure(figsize=(15, 11))

plt.subplot(2, 2, 1)
plt.plot(gt_x, gt_y, 'g-', linewidth=3, label='Ground Truth (RTK)')
plt.plot(base['est'][1], base['est'][2], '-', color='gray', linewidth=1.5, label='EKF без RTK')
plt.plot(main['est'][1], main['est'][2], 'b-', linewidth=2, label=f'EKF ({main_key})')
plt.plot(main['upd'][:, 0], main['upd'][:, 1], 'k.', markersize=6, label='Точки update')

plt.xlabel('X (м)')
plt.ylabel('Y (м)')
plt.title(f'Сравнение траекторий')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.subplot(2, 2, 2)
plt.plot((base['err_t'] - t0) / 1e9, base['error'], '-', color='gray', linewidth=1.5,
         label=f"Без RTK (средняя: {base['stats']['mean']:.3f} м)")
plt.plot((main['err_t'] - t0) / 1e9, main['error'], 'r-', linewidth=2,
         label=f"{main_key} (средняя: {main['stats']['mean']:.3f} м)")
plt.xlabel('Время (с)')
plt.ylabel('Ошибка (м)')
plt.title(f"Ошибка EKF во времени\n(макс: {main['stats']['max']:.3f} м, P95: {main['stats']['p95']:.3f} м)")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
x_pos = np.arange(len(N_LIST))
for name, label, style in [('mean', 'Средняя', 'o-'), ('median', 'Медиана', 's-'),
                           ('p95', 'P95', '^-'), ('max', 'Максимальная', 'v-')]:
    vals = [results[f'N={n}']['stats'][name] for n in N_LIST]
    plt.plot(x_pos, vals, style, linewidth=2, label=label)
plt.axhline(base['stats']['mean'], color='gray', linestyle='--', label='Средняя без RTK')
plt.xticks(x_pos, [f'N={n}\n({n / rtk_rate:.1f} с)' for n in N_LIST])
plt.xlabel('Прореживание RTK (период между update)')
plt.ylabel('Ошибка (м)')
plt.title('Зависимость ошибки от частоты RTK')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.hist(main['error'], bins=50, alpha=0.7, color='r', label=f'EKF {main_key}')
plt.hist(base['error'], bins=50, alpha=0.4, color='gray', label='EKF без RTK')
plt.axvline(main['stats']['mean'], color='r', linestyle='--')
plt.axvline(main['stats']['median'], color='k', linestyle=':')
plt.xlabel('Ошибка (м)')
plt.ylabel('Частота')
plt.title('Распределение ошибок')
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.savefig(Path("results") / f'citrus_ekf_res.png', dpi=150)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
for key, r in results.items():
    if key == 'Без RTK':
        continue
    plt.plot((r['err_t'] - t0) / 1e9, r['error'], '-', color=colors[key], linewidth=1.2,
             label=f"{key} (средняя: {r['stats']['mean']:.3f} м)")
plt.xlabel('Время (с)')
plt.ylabel('Ошибка (м)')
plt.title('Ошибка во времени при разной частоте RTK')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
keys = [k for k in results if k != 'Без RTK']
plt.boxplot([results[k]['error'] for k in keys], label=keys, showfliers=False)
plt.ylabel('Ошибка (м)')
plt.title('Распределение ошибки по режимам')
plt.grid(True)

plt.tight_layout()
# plt.savefig(Path("results") / f'citrus_ekf_sweep.png', dpi=150)
plt.show()