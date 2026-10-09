"""Навчання моделі: підгонка калібрувальних констант методом максимальної
правдоподібності. Потрібні лише numpy і pandas.

Запуск:  python train.py measured.csv shots.csv submission/weights.json

Ідея. Симулятор розігрує початок зриву як пуассонівський процес з інтенсивністю
lambda(t). Логарифм правдоподібності набору розрядів:
    sum_по_зривах log lambda(t_D)  -  sum_по_всіх_кроках lambda(t) * dt
Формула lambda відома (розд. 4.6 методички); невідомо лише, як прилади
переводяться у фізичні величини. Ці чотири константи й підбираємо:
    k_ne   - масштаб густини (уставка -> справжня густина)
    k_rad  - масштаб суми болометрів (-> P_rad, МВт)
    q_a, q_b - q95 = q_a * Bt/Ip * (1 + q_b * (I_vert/Ip)^2)
"""
import json
import sys
import time

import numpy as np
import pandas as pd

DT = 0.005
A_MINOR = 0.95
MU0 = 4e-7 * np.pi
BOLO = ["bolo_%02d" % i for i in range(1, 9)]


def ema(x, tau):
    """Причинне експоненційне згладжування зі сталою часу tau."""
    return pd.Series(x).ewm(alpha=DT / tau, adjust=False).mean().values


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def prepare(measured, shots):
    """Для кожного розряду - сирі фізичні величини до калібрування."""
    data = []
    for sid, d in measured.groupby("shot_id"):
        row = shots.loc[sid]
        d = d.ffill().bfill()
        ip = d["prog_ip_MA"].values
        bt = d["prog_bt_T"].values
        ip_c = np.maximum(ip, 0.05)
        t = d["t"].values
        f_gw = ema(d["prog_ne_1e20"].values, 0.3) / (ip_c / (np.pi * A_MINOR ** 2))
        beta_t = 100 * (2 / 3 * d["mag_diamag_MJ"].values * 1e6 / 100) / (bt ** 2 / (2 * MU0))
        p_aux = (d["prog_p_nbi_MW"] + d["prog_p_ich_MW"] + d["prog_p_ech_MW"]).values
        p_ohm = np.clip(ema(d["mag_vloop_V"].values, 0.1) * ip, 0, 5)
        t_d = row["t_trigger_s"] if row["disrupted"] else np.inf
        live = (ip > 0.24) & (ip >= 0.9 * np.maximum.accumulate(ip)) & (t <= t_d + 1e-6) & (t > 1.0)
        event = np.zeros(len(t), bool)
        if row["disrupted"]:
            event[np.argmin(np.abs(t - t_d))] = True
        data.append(dict(
            f_gw=f_gw, beta_n=ema(beta_t * A_MINOR * bt / ip_c, 0.1), bt_ip=bt / ip_c,
            kappa=ema(d["coil_pf_vert_kA"].values / ip_c, 0.05),
            p_rad=d[BOLO].sum(axis=1).values, p_in=p_aux + p_ohm, live=live, event=event))
    return data


def hazard(s, p):
    k_ne, k_rad, q_a, q_b = p
    f_gw = ema(s["f_gw"] * k_ne, 0.1)
    f_rad = ema(np.where(s["p_in"] > 1, k_rad * s["p_rad"] / np.maximum(s["p_in"], 1), 0), 0.1)
    q95 = q_a * s["bt_ip"] * (1 + q_b * s["kappa"] ** 2)
    r_gw = sigmoid((f_gw - 1.0) / 0.08)
    r_b = sigmoid((s["beta_n"] / 2.8 - 0.85) / 0.06)
    r_q = sigmoid((2.2 / q95 - 0.9) / 0.05)
    r_rad = sigmoid((f_rad - 0.8) / 0.08)
    r_lm = 5e-4 * (1 + 5 * r_gw + 3 * r_b + 2 * r_q)
    lam = 1 - (1 - r_gw ** 3) * (1 - r_b ** 3) * (1 - r_q ** 3) * (1 - r_rad ** 3) * (1 - r_lm)
    return np.maximum(lam, 0.005 + np.minimum(r_lm, 0.01))


def neg_log_likelihood(p, data):
    total = 0.0
    for s in data:
        lam = hazard(s, p)
        live = s["live"]
        total += lam[live].sum() * DT - np.log(lam[s["event"] & live]).sum()
    return total


def nelder_mead(f, x0, step=0.15, iters=300, tol=1e-4):
    """Симплекс-пошук мінімуму без похідних (щоб не залежати від scipy)."""
    n = len(x0)
    pts = [np.array(x0, float)]
    for i in range(n):
        x = np.array(x0, float)
        x[i] *= 1 + step
        pts.append(x)
    vals = [f(x) for x in pts]
    for it in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if it % 20 == 0:
            print("  ітерація %3d   -logL = %.4f   %s" % (it, vals[0], np.round(pts[0], 4)), flush=True)
        if vals[-1] - vals[0] < tol:
            break
        centre = np.mean(pts[:-1], axis=0)
        refl = centre + (centre - pts[-1])
        f_refl = f(refl)
        if f_refl < vals[0]:
            expd = centre + 2 * (centre - pts[-1])
            f_expd = f(expd)
            pts[-1], vals[-1] = (expd, f_expd) if f_expd < f_refl else (refl, f_refl)
        elif f_refl < vals[-2]:
            pts[-1], vals[-1] = refl, f_refl
        else:
            contr = centre + 0.5 * (pts[-1] - centre)
            f_contr = f(contr)
            if f_contr < vals[-1]:
                pts[-1], vals[-1] = contr, f_contr
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (x - pts[0]) for x in pts[1:]]
                vals = [vals[0]] + [f(x) for x in pts[1:]]
    best = int(np.argmin(vals))
    return pts[best], vals[best]


if __name__ == "__main__":
    measured_path, shots_path, weights_path = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.time()
    print("читаю дані...", flush=True)
    measured = pd.read_csv(measured_path)
    shots = pd.read_csv(shots_path).set_index("shot_id")
    data = prepare(measured, shots)
    print("розрядів: %d, зривів: %d  (%.0f с)" % (len(data), int(shots["disrupted"].sum()), time.time() - t0), flush=True)
    start = [1.0, 0.7, 1.5, 0.5]
    print("старт: -logL = %.4f" % neg_log_likelihood(start, data), flush=True)
    best, val = nelder_mead(lambda p: neg_log_likelihood(p, data), start)
    print("результат: k_ne=%.4f  k_rad=%.4f  q_a=%.4f  q_b=%.4f   -logL = %.4f" % (*best, val))
    with open(weights_path) as f:
        w = json.load(f)
    w.update(k_ne=round(float(best[0]), 4), k_rad=round(float(best[1]), 4),
             q_a=round(float(best[2]), 4), q_b=round(float(best[3]), 4))
    with open(weights_path, "w") as f:
        json.dump(w, f, indent=1)
    print("ваги записано в %s  (усього %.0f с)" % (weights_path, time.time() - t0))
