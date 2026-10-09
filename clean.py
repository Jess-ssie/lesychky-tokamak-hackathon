"""Очистка вимірюваних каналів токамака. Усе причинне: значення на кроці t
залежить лише від минулого, тож ту саму очистку можна робити й на заліку.

Три дії:
  1. Відмова приладу (порожня клітинка / nan) -> останнє справне значення.
  2. Шум -> експоненційне згладжування зі сталою часу, що залежить від приладу.
  3. Швидкі канали (ELM, МГД, locked mode) та уставки без шуму не чіпаємо.

Пакетний режим:   python clean.py measured.csv measured_clean.csv
У моделі:         c = Cleaner(); c.reset(); x = c.step(channels)
"""
import math
import sys

DT = 0.005

# стала часу згладжування за префіксом імені каналу, с (0 = не згладжувати)
TAU_BY_PREFIX = [
    ("prog_", 0.0),              # уставки програми: шуму немає
    ("gas_valve_", 0.0),         # команди клапанів: шуму немає
    ("mag_mirnov_", 0.0),        # МГД-коливання і ELM: швидкий сигнал і є дані
    ("mag_saddle_", 0.0),        # locked mode: передвісник живе 50-100 мс
    ("mag_halo_", 0.0),          # гало-струми: лише під час зриву
    ("dalpha_", 0.0),            # піки ELM тривають один-два кроки
    ("mag_ip_", 0.02), ("mag_bt_", 0.02), ("mag_flux_loop_", 0.02),
    ("coil_pf_", 0.02), ("coil_cs_", 0.02),
    ("coil_tf_temp", 0.5), ("coil_tf_", 0.05),
    ("mag_diamag_", 0.05), ("mag_vloop_", 0.05),
    ("ifr_", 0.05), ("ece_", 0.05), ("bolo_", 0.05), ("sxr_", 0.05),
    ("neutron_", 0.05), ("div_probe_", 0.05), ("gas_flow_", 0.05),
    ("nbi_beam_energy", 0.5), ("nbi_", 0.05), ("ich_", 0.05), ("ech_", 0.05),
    ("div_tc_", 0.5), ("wall_tc_", 0.5),      # термопари: повільна фізика
    ("vac_", 0.5), ("cryo_", 0.5), ("psu_", 0.5),
]


def tau_of(name):
    for prefix, tau in TAU_BY_PREFIX:
        if name.startswith(prefix):
            return tau
    return 0.05


class Cleaner:
    """Покрокова очистка для використання всередині моделі."""

    def __init__(self, dt=DT):
        self.dt = dt
        self.reset()

    def reset(self):
        self.last = {}      # останнє справне сире значення
        self.smooth = {}    # згладжене значення
        self.missing = 0    # скільки каналів мовчить на цьому кроці

    def step(self, channels):
        out = {}
        self.missing = 0
        for name, v in channels.items():
            if v is None or v != v:              # nan != nan
                self.missing += 1
                v = self.last.get(name)
                if v is None:                    # прилад мовчить від самого початку
                    out[name] = math.nan
                    continue
            else:
                self.last[name] = v
            tau = tau_of(name)
            if tau > 0 and name in self.smooth:
                v = self.smooth[name] + self.dt / tau * (v - self.smooth[name])
            self.smooth[name] = v
            out[name] = v
        return out


def clean_frame(df, dt=DT):
    """Та сама очистка для цілого measured.csv (pandas), розряд за розрядом."""
    import pandas as pd
    chans = [c for c in df.columns if c not in ("shot_id", "t", "p")]
    out = df.copy()
    out["n_missing"] = df[chans].isna().sum(axis=1)
    grouped = df.groupby("shot_id", sort=False)
    filled = grouped[chans].ffill()              # крок 1: тримаємо останнє значення
    for c in chans:
        tau = tau_of(c)
        if tau > 0:                              # крок 2: згладжування
            alpha = dt / tau
            filled[c] = filled[c].groupby(df["shot_id"], sort=False).transform(
                lambda s: s.ewm(alpha=alpha, adjust=False, ignore_na=True).mean())
    out[chans] = filled
    return out


if __name__ == "__main__":
    import pandas as pd
    src, dst = sys.argv[1], sys.argv[2]
    frame = pd.read_csv(src)
    cleaned = clean_frame(frame)
    cleaned.to_csv(dst, index=False, float_format="%.6g")
    print("рядків: %d, заповнено пропусків: %d, рядків із пропусками: %d" % (
        len(frame), int(frame.drop(columns=["shot_id", "t"]).isna().sum().sum()),
        int((cleaned["n_missing"] > 0).sum())))
