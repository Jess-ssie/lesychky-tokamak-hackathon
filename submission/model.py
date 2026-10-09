"""Раннє попередження зриву токамака.

Кожен крок: канали -> стан плазми -> відстань до чотирьох меж ->
інтенсивність зриву lambda (формула симулятора, розд. 4.6 методички) ->
очікуваний інтеграл lambda до кінця небезпечної фази -> імовірність p.

Три джерела ризику складаються в один показник L (очікувана кількість
"жеребів, що випали"), і p = 1 - exp(-L):
  1. поточний стан: lambda зараз * час до спаду струму;
  2. прогноз на 0.5 с: уставки густини й струму випереджають плазму;
  3. неон: якщо клапан неону відкривали сильно, після вимкнення нагріву
     частка випромінювання перевищить межу.
Окремо - детектор locked mode для останніх десятків мілісекунд.
"""
import json
import math
import os
from collections import deque

A_MINOR = 0.95          # малий радіус JET, м
VOLUME = 100.0          # об'єм плазми, м^3
MU0 = 4e-7 * math.pi
GW_AREA = math.pi * A_MINOR ** 2
BOLO = ["bolo_%02d" % i for i in range(1, 9)]
CHORDS = [("ifr_chord_core", 1.0), ("ifr_chord_mid_in", 0.863), ("ifr_chord_mid_out", 0.859)]
TAU_NE = 0.3            # густина наздоганяє уставку з цією сталою часу, с


def sigmoid(x):
    if x < -40.0:
        return 0.0
    if x > 40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


class Model:
    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json")
        with open(weights_path, encoding="utf-8") as f:
            self.w = json.load(f)
        self.start_shot({})

    # ---------- новий розряд: скидаємо всю пам'ять ----------
    def start_shot(self, meta):
        self.dt = float(meta.get("dt", 0.005) or 0.005)
        dur = meta.get("duration")
        self.duration = float(dur) if dur else None
        self.names = meta.get("channels") or []
        self.last = {}          # останнє справне значення кожного каналу
        self.ip_peak = 0.0      # найбільша досягнута уставка струму
        self.ne = None          # густина, що наздоганяє уставку
        self.ne_meas = None     # густина з інтерферометра
        self.f_gw = None        # згладжені входи сигмоїд (tau = 0.1 c, як у симуляторі)
        self.f_rad = None
        self.beta_n = None
        self.vloop = None
        self.kappa = None
        self.lm = 0.0           # згладжений сигнал сідлових котушок
        self.ip_prev = None     # попередня справна уставка струму
        self.t_flat = None      # момент виходу струму на плато
        self.neon_peak = 0.0    # найбільша команда клапана неону за розряд
        n = max(int(round(self.w["slope_window"] / self.dt)), 1)
        self.hist_ne = deque(maxlen=n + 1)   # історія уставок для оцінки нахилу
        self.hist_ip = deque(maxlen=n + 1)
        self.state = {}

    # канал із пам'яттю: nan/None -> останнє справне значення
    def get(self, ch, name, default=0.0):
        v = ch.get(name)
        if v is None or v != v:
            return self.last.get(name, default)
        self.last[name] = v
        return v

    def ema(self, old, new, tau):
        if old is None:
            return new
        return old + self.dt / tau * (new - old)

    # ---------- формула ризику симулятора ----------
    def hazard(self, f_gw, beta_n, q95, f_rad):
        w = self.w
        r_gw = sigmoid((f_gw - w["c_gw"]) / w["w_gw"])
        r_b = sigmoid((beta_n / 2.8 - w["c_beta"]) / w["w_beta"])
        r_q = sigmoid((2.2 / q95 - w["c_q"]) / w["w_q"])
        r_rad = sigmoid((f_rad - w["c_rad"]) / w["w_rad"])
        r_lm = 5e-4 * (1 + 5 * r_gw + 3 * r_b + 2 * r_q)
        lam = 1 - (1 - r_gw ** 3) * (1 - r_b ** 3) * (1 - r_q ** 3) * (1 - r_rad ** 3) * (1 - r_lm)
        return max(lam, 0.005 + min(r_lm, 0.01))

    def predict(self, t, ch):
        w = self.w
        dt = self.dt
        g = self.get
        if not isinstance(ch, dict):                      # якщо прийшов список, а не словник
            ch = dict(zip(self.names, ch))

        # --- 1. стан плазми з приладів ---
        ip = g(ch, "prog_ip_MA", g(ch, "mag_ip_MA"))      # уставка струму: без шуму
        bt = g(ch, "prog_bt_T", g(ch, "mag_bt_T", 3.0))
        ne_set = g(ch, "prog_ne_1e20", None)
        # потужність нагріву: уставка, а якщо її канал мовчить від початку - вимір
        p_aux = (g(ch, "prog_p_nbi_MW", g(ch, "nbi_power_MW"))
                 + g(ch, "prog_p_ich_MW", g(ch, "ich_power_MW"))
                 + g(ch, "prog_p_ech_MW", g(ch, "ech_power_MW")))
        ip_c = max(ip, 0.05)
        self.ip_peak = max(self.ip_peak, ip)
        if ip <= 0.24:          # плазми ще немає: згладжені входи не накопичуємо
            self.f_gw = self.f_rad = self.beta_n = None

        # густина: релаксує до уставки; інтерферометр - контроль зверху
        chord = 0.0
        for name, k in CHORDS:
            chord += g(ch, name) / k
        self.ne_meas = self.ema(self.ne_meas, chord / 3.0 / w["chord_len"], 0.05)
        if ne_set is None:      # уставки густини немає: лишається інтерферометр
            ne = self.ne_meas
            ne_set = self.ne_meas
        else:
            self.ne = self.ema(self.ne, ne_set, TAU_NE)
            ne = min(self.ne * w["k_ne"], w["meas_cap"] * self.ne_meas + 0.02)
        self.f_gw = self.ema(self.f_gw, ne / (ip_c / GW_AREA), 0.1)

        # бета: з діамагнітної петлі
        w_th = g(ch, "mag_diamag_MJ")
        beta_t = 100.0 * (2.0 / 3.0 * w_th * 1e6 / VOLUME) / (bt * bt / (2 * MU0))
        self.beta_n = self.ema(self.beta_n, beta_t * A_MINOR * bt / ip_c, 0.1)

        # q95: Bt/Ip і витягнутість (котушка вертикального поля / струм)
        self.kappa = self.ema(self.kappa, g(ch, "coil_pf_vert_kA") / ip_c, 0.05)
        q95 = w["q_a"] * bt / ip_c * (1 + w["q_b"] * self.kappa * self.kappa)

        # частка випромінювання: сума болометрів / (нагрів + омічний)
        self.vloop = self.ema(self.vloop, g(ch, "mag_vloop_V"), 0.1)
        p_ohm = min(max(self.vloop * ip, 0.0), 5.0)
        p_in = p_aux + p_ohm
        p_rad = 0.0
        for name in BOLO:
            p_rad += g(ch, name)
        self.f_rad = self.ema(self.f_rad, p_rad / p_in if p_in > 1.0 else 0.0, 0.1)
        f_rad = w["k_rad"] * self.f_rad

        # --- 2. інтенсивність зриву зараз ---
        # на спаді струму жереб не кидається; першу секунду не довіряємо оцінкам
        live = ip > 0.24 and ip >= 0.9 * self.ip_peak and t > w["t_min"]
        lam = self.hazard(self.f_gw, self.beta_n, q95, f_rad) if live else 0.0

        # --- 3. скільки ще триватиме небезпечна фаза ---
        # програма має сталу форму: плато струму починається на 0.15 тривалості,
        # спад - на 0.80, жереб вимикається на 0.82 (струм нижче 90 % піку)
        raw_ip = ch.get("prog_ip_MA")
        ok = raw_ip is not None and raw_ip == raw_ip
        slope_now = 0.0
        if ok and self.ip_prev is not None:
            slope_now = (raw_ip - self.ip_prev) / dt
            if self.t_flat is None and raw_ip > 0.24 and slope_now <= 1e-7 and t > 0.02:
                self.t_flat = t - dt
        if ok:
            self.ip_prev = raw_ip
        dur = self.t_flat / 0.15 if self.t_flat else self.duration
        if dur:
            t_rem = max(w["gate_frac"] * dur - t, 0.0)
            t_frac = t / dur
        else:
            t_rem = 1.0
            t_frac = 0.0
        if live and slope_now < -1e-6 and ip < self.ip_peak:      # спад уже йде
            t_rem = min(t_rem, (ip - 0.9 * self.ip_peak) / -slope_now)
        t_rem = min(max(t_rem, 0.05), w["h_max"])
        risk = lam * t_rem

        # --- 4. прогноз на fc_h секунд: уставки випереджають плазму ---
        # (а) плазма напевно дійде до ПОТОЧНОЇ уставки густини - вага 1;
        # (б) уставки можуть і далі рости з тим самим нахилом - це лише
        #     припущення, тому вага fc_slope_weight < 1
        self.hist_ne.append(ne_set)
        self.hist_ip.append(ip)
        if live and len(self.hist_ne) == self.hist_ne.maxlen:
            span = w["slope_window"]
            h = w["fc_h"]
            decay = math.exp(-h / TAU_NE)
            ne_lag = self.ne if self.ne is not None else ne_set
            left = max(t_rem - h, 0.0)
            ne_a = ne_set - (ne_set - ne_lag) * decay
            lam_a = self.hazard(w["k_ne"] * ne_a / (ip_c / GW_AREA), self.beta_n, q95, f_rad)
            risk = max(risk, lam_a * left)
            s_ne = (ne_set - self.hist_ne[0]) / span
            s_ip = max((ip - self.hist_ip[0]) / span, 0.0) if t_frac > 0.16 else 0.0
            # розв'язок рівняння релаксації при лінійній уставці
            ne_b = ne_set + s_ne * h - TAU_NE * s_ne - (ne_set - TAU_NE * s_ne - ne_lag) * decay
            ip_b = ip_c + s_ip * h
            lam_b = self.hazard(w["k_ne"] * ne_b / (ip_b / GW_AREA), self.beta_n, q95 * ip_c / ip_b, f_rad)
            risk = max(risk, w["fc_slope_weight"] * lam_b * left)

        # --- 5. неон: ризик у вікні після вимкнення нагріву ---
        self.neon_peak = max(self.neon_peak, g(ch, "gas_valve_ne_cmd"))
        p_neon = 0.0
        if live and t_frac < w["neon_until"]:
            p_neon = w["neon_p"] * sigmoid((self.neon_peak - w["neon_thr"]) / w["neon_w"])
            risk += -math.log(1.0 - p_neon)

        # --- 6. передвісник: дрейф сідлових котушок (locked mode) ---
        self.lm = self.ema(self.lm, g(ch, "mag_saddle_lm"), w["lm_tau"])
        p_lm = sigmoid((self.lm - w["lm_thr"]) / w["lm_w"]) if ip > 0.24 else 0.0

        p = max(1.0 - math.exp(-risk), p_lm)
        self.state = dict(t=t, lam=lam, f_gw=self.f_gw, beta_n=self.beta_n, q95=q95, f_rad=f_rad,
                          t_rem=t_rem, p_neon=p_neon, p_lm=p_lm, live=live, dur=dur or 0.0)
        if p != p:
            return 0.0
        return min(max(p, 0.0), 1.0)


if __name__ == "__main__":
    from toksym_model import serve
    serve(Model())
