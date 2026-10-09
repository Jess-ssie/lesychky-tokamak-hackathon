"""Власна перевірка моделі на будь-якому наборі.
Запуск:  python evaluate.py measured.csv shots.csv submission
Друкує таблицю TPR (частка спійманих зривів) для сітки Lead time x FPR
і середнє по таблиці - наближення до фінального "об'єму під поверхнею".
Спонтанні зриви (shots.csv: spontaneous=true) виключено, як у заліку.
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

LEADS = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
FPRS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]


def main(measured, shots, sub):
    spec = importlib.util.spec_from_file_location("model", os.path.join(sub, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.Model()
    df = pd.read_csv(measured)
    sh = pd.read_csv(shots).set_index("shot_id")
    cols = list(df.columns[2:])
    out, times = {}, []
    for sid, d in df.groupby("shot_id"):
        model.start_shot({"shot_id": int(sid), "dt": 0.005})
        vals, ts = d[cols].values, d["t"].values
        p = np.empty(len(ts))
        for i in range(len(ts)):
            t0 = time.perf_counter()
            p[i] = model.predict(float(ts[i]), dict(zip(cols, vals[i].tolist())))
            times.append(time.perf_counter() - t0)
        out[sid] = (ts, p)
    print("крок моделі, мс: p50 %.3f  p99 %.3f" % (np.percentile(times, 50) * 1e3, np.percentile(times, 99) * 1e3))
    safe = [s for s in out if not sh.loc[s, "disrupted"]]
    dis = [s for s in out if sh.loc[s, "disrupted"] and not sh.loc[s, "spontaneous"]]
    smax = np.sort([out[s][1].max() for s in safe])[::-1]
    tab = np.zeros((len(LEADS), len(FPRS)))
    for j, f in enumerate(FPRS):
        k = int(np.floor(f * len(safe) + 1e-9))       # дозволена кількість хибних тривог
        thr = smax[k] if k < len(smax) else -1.0      # тривога, коли p > thr
        for i, lead in enumerate(LEADS):
            caught = sum(np.any(out[s][1][out[s][0] <= sh.loc[s, "t_tq_s"] - lead + 1e-9] > thr) for s in dis)
            tab[i, j] = caught / max(len(dis), 1)
    print("безпечних: %d, зривів у заліку: %d" % (len(safe), len(dis)))
    print(pd.DataFrame(tab, index=["lead %.2f" % l for l in LEADS], columns=["FPR %.2f" % f for f in FPRS]).round(2).to_string())
    print("середній TPR по сітці: %.3f" % tab.mean())
    pd.DataFrame({"shot_id": list(out), "max_p": [out[s][1].max() for s in out]}).to_csv("max_p_by_shot.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "submission")
