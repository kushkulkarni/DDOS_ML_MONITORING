"""
Traffic + server + SDN-mitigation simulator for the DDoS demo.

Everything here is a *simulation of numbers* - no real packets are generated.
The ML verdict comes from the real ddos_model.pkl (RandomForest + StandardScaler).
"""
import math
import random
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

DT = 0.5                      # seconds of simulated time per tick
WINDOW_TICKS = 6              # detector looks at the last 3 s of traffic

# --- server -----------------------------------------------------------------
SERVER_CAPACITY = 1500.0      # requests/s the server can process
BACKLOG_MAX = 6000.0          # SYN / accept queue size
CRASH_AFTER_TICKS = 10        # sustained saturation (5 s) -> server goes down

# --- traffic ----------------------------------------------------------------
LEGIT_CLIENTS = 8
BOTS = 600
INTENSITY = {"low": 5000.0, "medium": 9000.0, "high": 18000.0}
ATTACK_RAMP_TICKS = 8         # attack climbs to full rate over 4 s
ATTACK_STOP_TICKS = 3

# --- adapter: simulator telemetry -> model training scale -------------------
PKTS_PER_LEGIT_REQ = 10
BYTES_PER_LEGIT_PKT = 1000.0
BYTES_PER_ATTACK_PKT = 60.0   # a bare SYN
COUNTER_SCALE = 67.0          # stretch a 3 s window to the capture length the model was trained on
FEATURE_CLIP = {"pktcount": 260_000, "tot_kbps": 20_000}
TRAIN_DURATION_NS = 3.0e11    # tot_dur in the training data is in nanoseconds

FEATURE_ORDER = ['pktcount', 'byteperflow', 'tot_kbps', 'rx_kbps', 'flows',
                 'bytecount', 'tot_dur', 'Protocol_ICMP', 'Protocol_TCP', 'Protocol_UDP']

ML_THRESHOLD = 0.5
CONFIRM_TICKS = 2             # model must say "attack" on 2 of the last 3 ticks
RATE_GUARD = 2.5              # ...and traffic must be >2.5x the learned baseline
CALM_TICKS_TO_CLEAR = 8


class Detector:
    """Wraps the real model; falls back to a simple rate rule if it can't load."""

    def __init__(self, model_path):
        self.model = self.scaler = None
        self.error = None
        self.path = str(model_path)
        try:
            import joblib
            import warnings
            warnings.filterwarnings("ignore")
            obj = joblib.load(model_path)
            if isinstance(obj, dict):
                self.model, self.scaler = obj["model"], obj.get("scaler")
            else:
                self.model = obj
        except Exception as e:  # noqa
            self.error = f"{type(e).__name__}: {e}"

    @property
    def loaded(self):
        return self.model is not None

    def predict_proba(self, feats: dict) -> float:
        import pandas as pd
        X = pd.DataFrame([feats], columns=FEATURE_ORDER)
        if self.scaler is not None:
            X = pd.DataFrame(self.scaler.transform(X), columns=FEATURE_ORDER)
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        return float(proba[classes.index(1)]) if 1 in classes else 0.0


class Simulation:
    def __init__(self, model_path):
        self.detector = Detector(model_path)
        self.lock = threading.Lock()
        self.rng = random.Random()
        self.auto_mitigation = True
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self):
        with getattr(self, "lock", threading.Lock()):
            self.tick = 0
            self.legit_rate = 70.0
            self.baseline = 70.0
            self.attack_on = False
            self.attack_target = 0.0
            self.attack_rate = 0.0
            self.intensity = "medium"
            self.stop_ticks_left = 0

            self.window = deque([(70.0, 0.0)] * WINDOW_TICKS, maxlen=WINDOW_TICKS)   # (legit, attack) raw rps, pre-filled with calm traffic
            self.p_attack = 0.05
            self.ml_hist = deque(maxlen=3)
            self.features = self._make_features()
            self.alert = False
            self.alert_tick = None
            self.calm = 0

            # mitigation
            self.stage_tick = {1: None, 2: None, 3: None}
            self.blocked_frac = 0.0
            self.admission_cap = None
            self.syn_cookies = False
            self.blocked_ips = 0
            self.dropped_total = 0
            self.stable_ticks = 0
            self.protected = False

            # server
            self.backlog = 0.0
            self.overload_ticks = 0
            self.crashed = False
            self.cpu = 8.0
            self.latency = 42.0
            self.success = 100.0
            self.delivered = 0.0
            self.legit_pass_rate = 0.0
            self.atk_pass_rate = 0.0
            self.blocked_rps = 0.0
            self.legit_served = 0.0

            self.history = deque(maxlen=160)
            self.logs = deque(maxlen=60)
            self.log_seq = 0
            self._log("info", "System online. Controller and ML model watching switch s1.")
            if self.detector.loaded:
                self._log("info", "Random Forest model loaded from " + Path(self.detector.path).name)
            else:
                self._log("warn", "Model not loaded (" + str(self.detector.error) + "). Using rate-rule fallback.")

    def _log(self, level, msg):
        self.log_seq += 1
        self.logs.append({"seq": self.log_seq, "level": level, "msg": msg,
                          "t": round(self.tick * DT, 1)})

    # --------------------------------------------------------------- controls
    def start_attack(self, intensity="medium"):
        with self.lock:
            if self.crashed or self.attack_on:
                return
            self.intensity = intensity if intensity in INTENSITY else "medium"
            self.attack_target = INTENSITY[self.intensity]
            self.attack_on = True
            self.stop_ticks_left = 0
            self._log("crit", f"Botnet of {BOTS} hosts started a SYN flood ({self.intensity} intensity).")

    def stop_attack(self):
        with self.lock:
            if not self.attack_on:
                return
            self.attack_on = False
            self.stop_ticks_left = ATTACK_STOP_TICKS
            self._log("info", "Attack traffic stopped.")

    def set_auto_mitigation(self, on: bool):
        with self.lock:
            self.auto_mitigation = bool(on)
            self._log("info", "Auto-mitigation " + ("enabled." if on else "DISABLED. Alerts will not trigger defences."))

    # ------------------------------------------------------------- one tick
    def step(self):
        with self.lock:
            self.tick += 1

            # --- traffic generation -----------------------------------------
            if not self.alert:
                # only learn the baseline while traffic looks normal
                self.baseline = 0.95 * self.baseline + 0.05 * self.legit_rate
            self.legit_rate = min(95.0, max(50.0, self.legit_rate + self.rng.uniform(-6, 6)))
            if self.attack_on:
                self.attack_rate = min(self.attack_target, self.attack_rate + self.attack_target / ATTACK_RAMP_TICKS)
            elif self.attack_rate > 0:
                self.attack_rate = max(0.0, self.attack_rate - INTENSITY["high"] / ATTACK_STOP_TICKS)
                if self.attack_rate < 5:
                    self.attack_rate = 0.0
            legit = self.legit_rate
            atk = self.attack_rate * (1 + self.rng.uniform(-0.03, 0.03)) if self.attack_rate else 0.0
            self.window.append((legit, atk))

            # --- detection ---------------------------------------------------
            self._detect(legit, atk)

            # --- mitigation --------------------------------------------------
            self._mitigate()

            # --- what the switch lets through ---------------------------------
            atk_pass = atk * (1 - self.blocked_frac)
            if self.syn_cookies:
                atk_pass *= 0.35
            legit_pass = legit * (0.985 if self.blocked_frac > 0 else 1.0)
            total_pass = legit_pass + atk_pass
            if self.admission_cap and total_pass > self.admission_cap:
                k = self.admission_cap / total_pass
                legit_pass *= k
                atk_pass *= k
                total_pass = self.admission_cap
            self.blocked_rps = max(0.0, (legit + atk) - total_pass)
            self.dropped_total += int(self.blocked_rps * DT)
            self.delivered = total_pass
            self.legit_pass_rate = legit_pass
            self.atk_pass_rate = atk_pass

            # --- server -------------------------------------------------------
            self._server(legit, legit_pass, total_pass)

            self.history.append({
                "t": round(self.tick * DT, 1),
                "incoming": round(legit + atk, 1),
                "delivered": round(self.delivered, 1),
                "p": round(self.p_attack, 3),
                "cpu": round(self.cpu, 1),
                "alert": int(self.alert),
            })

    # -------------------------------------------------------------- detector
    def _make_features(self):
        n = len(self.window)
        pk = by = 0.0
        for l, a in self.window:
            pk += (l * PKTS_PER_LEGIT_REQ + a) * DT
            by += (l * PKTS_PER_LEGIT_REQ * BYTES_PER_LEGIT_PKT + a * BYTES_PER_ATTACK_PKT) * DT
        bpp = by / max(pk, 1.0)
        pk_s = min(pk * COUNTER_SCALE, FEATURE_CLIP["pktcount"])
        by_s = pk_s * bpp
        avg_rps = sum(l + a for l, a in self.window) / n
        flows = int(min(12, max(1, round(2 + 1.5 * math.log10(1 + avg_rps)))))
        kbps = min(by * COUNTER_SCALE * 8 / 1000 / (n * DT), FEATURE_CLIP["tot_kbps"])
        return {
            'pktcount': pk_s, 'byteperflow': by_s / flows, 'tot_kbps': kbps,
            'rx_kbps': kbps * 0.5, 'flows': flows, 'bytecount': by_s,
            'tot_dur': TRAIN_DURATION_NS,
            'Protocol_ICMP': 0, 'Protocol_TCP': 1, 'Protocol_UDP': 0,
        }

    def _detect(self, legit, atk):
        self.features = self._make_features()
        raw = legit + atk
        ratio = raw / max(self.baseline, 1.0)
        if self.detector.loaded:
            try:
                self.p_attack = self.detector.predict_proba(self.features)
            except Exception as e:  # noqa
                self.p_attack = 0.0
                self.detector.error = str(e)
        else:
            self.p_attack = min(1.0, max(0.0, (ratio - 1) / 6))
        self.ratio = ratio
        self.ml_hist.append(self.p_attack >= ML_THRESHOLD)
        votes = sum(self.ml_hist)

        if not self.alert:
            if votes >= CONFIRM_TICKS and ratio >= RATE_GUARD:
                self.alert = True
                self.alert_tick = self.tick
                self.calm = 0
                self._log("crit", f"ML ALERT: DDoS detected. P(attack)={self.p_attack:.0%}, traffic {ratio:.0f}x normal.")
                if not self.auto_mitigation:
                    self._log("warn", "Auto-mitigation is off. Nothing is stopping the flood.")
        else:
            if ratio < 1.8 and self.p_attack < ML_THRESHOLD:
                self.calm += 1
            else:
                self.calm = 0
            if self.calm >= CALM_TICKS_TO_CLEAR:
                self._log("ok", "Traffic back to normal. Releasing blacklist and rate limits.")
                self.alert = False
                self.alert_tick = None
                self.stage_tick = {1: None, 2: None, 3: None}
                self.blocked_frac = 0.0
                self.admission_cap = None
                self.syn_cookies = False
                self.blocked_ips = 0
                self.protected = False
                self.stable_ticks = 0
                self.ml_hist.clear()
                self.baseline = max(self.legit_rate, 50.0)

    # ------------------------------------------------------------ mitigation
    def _mitigate(self):
        if not (self.alert and self.auto_mitigation):
            return
        since = self.tick - self.alert_tick
        if since >= 1 and self.stage_tick[1] is None:
            self.stage_tick[1] = self.tick
            self.admission_cap = SERVER_CAPACITY * 0.85
            self._log("warn", f"Stage 1: OpenFlow meter installed. Admission capped at {int(self.admission_cap)} req/s to protect the server.")
        if since >= 3 and self.stage_tick[2] is None:
            self.stage_tick[2] = self.tick
            self._log("warn", "Stage 2: blacklisting top-talker sources with DROP flow rules.")
        if self.stage_tick[2] is not None:
            self.blocked_frac = min(0.985, self.blocked_frac + 0.14)
            self.blocked_ips = int(BOTS * self.blocked_frac)
        if since >= 6 and self.stage_tick[3] is None:
            self.stage_tick[3] = self.tick
            self.syn_cookies = True
            self._log("warn", "Stage 3: SYN cookies on. Remaining half-open connections are validated.")

    # ---------------------------------------------------------------- server
    def _server(self, legit_in, legit_pass, arrivals):
        if self.crashed:
            self.cpu, self.latency, self.success, self.legit_served = 0.0, 0.0, 0.0, 0.0
            self.backlog = BACKLOG_MAX
            return
        processed = min(arrivals, SERVER_CAPACITY)
        self.backlog += (arrivals - processed) * DT
        if arrivals <= SERVER_CAPACITY:
            self.backlog *= 0.55
        self.backlog = min(BACKLOG_MAX, max(0.0, self.backlog))
        self.cpu = min(100.0, arrivals / SERVER_CAPACITY * 100 + self.rng.uniform(-2, 2) + 5)
        self.cpu = max(3.0, self.cpu)
        self.latency = 40 + (self.backlog / SERVER_CAPACITY) * 1000
        share = min(1.0, SERVER_CAPACITY / arrivals) if arrivals > 0 else 1.0
        if self.backlog >= BACKLOG_MAX * 0.98:
            share *= 0.8
        self.legit_served = legit_pass * share
        self.success = max(0.0, min(100.0, self.legit_served / max(legit_in, 1.0) * 100))

        saturated = arrivals >= SERVER_CAPACITY and self.backlog >= BACKLOG_MAX * 0.9
        self.overload_ticks = self.overload_ticks + 1 if saturated else max(0, self.overload_ticks - 1)
        if self.overload_ticks >= CRASH_AFTER_TICKS:
            self.crashed = True
            self._log("crit", "SERVER CRASHED. Connection queue overflowed and the service stopped responding.")

        if self.alert and self.auto_mitigation and not self.crashed:
            if self.cpu < 75 and self.success > 85:
                self.stable_ticks += 1
            else:
                self.stable_ticks = 0
            if self.stable_ticks >= 4 and not self.protected:
                self.protected = True
                self._log("ok", f"Server stable under attack: CPU {self.cpu:.0f}%, {self.success:.0f}% of real users served.")

    # -------------------------------------------------------------- snapshot
    def phase(self):
        if self.crashed:
            return "crashed"
        if self.alert:
            if not self.attack_on and self.attack_rate == 0:
                return "recovering"
            if not self.auto_mitigation:
                return "detected"
            if self.protected:
                return "protected"
            return "mitigating" if self.stage_tick[1] is not None else "detected"
        if self.attack_on or self.attack_rate > 0:
            return "attacking"
        return "normal"

    def snapshot(self):
        with self.lock:
            legit = self.legit_rate
            atk = self.attack_rate
            ph = self.phase()
            if ph == "crashed":
                status = "down"
            elif self.cpu > 90 or self.success < 70:
                status = "overloaded"
            elif self.cpu > 70 or self.success < 92:
                status = "degraded"
            else:
                status = "online"
            return {
                "tick": self.tick, "time": round(self.tick * DT, 1), "phase": ph,
                "auto_mitigation": self.auto_mitigation,
                "attack_on": self.attack_on, "intensity": self.intensity,
                "traffic": {
                    "legit": round(legit, 1), "attack": round(atk, 1),
                    "total": round(legit + atk, 1), "delivered": round(self.delivered, 1),
                    "legit_delivered": round(self.legit_pass_rate, 1), "attack_delivered": round(self.atk_pass_rate, 1),
                    "blocked": round(self.blocked_rps, 1), "bots": BOTS,
                    "blocked_ips": self.blocked_ips, "dropped_total": self.dropped_total,
                    "baseline": round(self.baseline, 1),
                },
                "server": {
                    "status": status, "cpu": round(self.cpu, 1), "latency": round(self.latency),
                    "backlog_pct": round(self.backlog / BACKLOG_MAX * 100, 1),
                    "success": round(self.success, 1), "capacity": SERVER_CAPACITY,
                },
                "ml": {
                    "loaded": self.detector.loaded, "error": self.detector.error,
                    "p_attack": round(self.p_attack, 3), "threshold": ML_THRESHOLD,
                    "votes": int(sum(self.ml_hist)), "confirm": CONFIRM_TICKS,
                    "ratio": round(getattr(self, "ratio", 1.0), 1), "rate_guard": RATE_GUARD,
                    "alert": self.alert,
                    "features": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.features.items()},
                },
                "mitigation": {
                    "stages": [
                        {"id": 1, "name": "Rate-limit the switch", "detail": "OpenFlow meter caps traffic below server capacity",
                         "active": self.stage_tick[1] is not None},
                        {"id": 2, "name": "Blacklist attacking sources", "detail": "DROP rules for top-talker IPs on s1",
                         "active": self.stage_tick[2] is not None},
                        {"id": 3, "name": "Validate connections", "detail": "SYN cookies filter spoofed handshakes",
                         "active": self.stage_tick[3] is not None},
                    ],
                    "blocked_frac": round(self.blocked_frac, 3),
                    "admission_cap": self.admission_cap,
                },
                "history": list(self.history),
                "logs": list(self.logs),
            }


class Runner(threading.Thread):
    def __init__(self, sim):
        super().__init__(daemon=True)
        self.sim = sim

    def run(self):
        while True:
            t0 = time.time()
            try:
                self.sim.step()
            except Exception as e:  # noqa
                print("sim error:", e)
            time.sleep(max(0.0, DT - (time.time() - t0)))
