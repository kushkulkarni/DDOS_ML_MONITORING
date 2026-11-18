# DDoS Shield: live detection and mitigation demo

A browser demo of the full workflow: normal traffic, press **Launch DDoS attack**, request count climbs,
your `ddos_model.pkl` flags it, and the "controller" defends the server so it never crashes.

All traffic is simulated numbers. No real packets are sent anywhere.

## Run it

Copy these files into your `CP` folder (next to `ddos_model.pkl`):

```
app.py   sim.py   static/index.html   requirements.txt
```

```bash
pip install flask scikit-learn pandas numpy joblib
python3 app.py
# open http://127.0.0.1:5000
```

Model elsewhere? `DDOS_MODEL=/path/to/ddos_model.pkl python3 app.py`
Different port? `PORT=8080 python3 app.py`

## What happens, step by step

| Stage | What you see |
|---|---|
| 1. Normal traffic | ~70 req/s from 8 users. CPU ~10%. ML probability near 0-5%. |
| 2. Attack ramps up | Botnet of 600 hosts ramps to 5k / 9k / 18k req/s over 4 s. Counter climbs, server overloads. |
| 3. ML detects it | Random Forest probability crosses 50%. Alert fires when 2 of the last 3 readings say "attack" AND traffic is >2.5x baseline. |
| 4. Defences engage | Stage 1 rate-limit (protects server within ~1 s), stage 2 blacklist attacking sources, stage 3 SYN cookies. |
| 5. Server stays online | CPU back to ~10-20%, ~98% of real users still served. Stop the attack and it releases the blocks. |

Toggle **Auto-mitigation** off and relaunch to show the unprotected case: the connection queue overflows and the
server goes down after about 5 seconds.

Measured with your real model (headless runs): 0 crashes with mitigation on, 0 false alarms in 400 s of normal
traffic, detection 2 to 4.5 s after launch, server stable about 9 s after launch.

## How the real model is used

`sim.py` turns the simulated traffic of the last 3 s into the 10 features your model was trained on
(`pktcount, byteperflow, tot_kbps, rx_kbps, flows, bytecount, tot_dur, Protocol_*`) and calls
`predict_proba` on the unmodified `ddos_model.pkl` + scaler. The attack is a SYN flood: many tiny (60 B)
packets, which is the pattern this model separates from normal web traffic. The "2 of 3 readings + rate guard"
rule exists because the forest alone is noisy on small windows.

Tuning knobs are constants at the top of `sim.py` (server capacity, bot count, intensities, thresholds).

## Things I noticed in the existing lab files

Found while reading the project. The first two I confirmed by running the model, the rest by reading the code
(Ryu is not installed here, so I did not run the controller).

1. **`tot_dur` unit mismatch.** The scaler's mean for `tot_dur` is about 3.2e11, so training used nanoseconds.
   `ddos_controller.py` and `test_pipeline.py` pass seconds (e.g. `5.0`), which is far outside the training range.
2. **`test_pipeline.py` "normal" sample is predicted as an attack** (about 82% attack). The model is very sensitive
   to feature scale, so real switch counters will not map cleanly onto it without matching how the dataset was built.
3. **`ddos_controller.py`: `self.datapaths` is never created**, so the first poll raises `AttributeError`.
4. `OFPFlowStatsRequest(..., ofproto.OFP_MATCH_TYPE_EXACT)` does not look like a valid argument for OpenFlow 1.3.
5. The controller only installs a table-miss rule to the controller and never forwards packets, so `pingall`
   in Mininet will fail until a learning-switch or static forwarding rules are added.
6. `Protocol_ICMP/TCP/UDP` are hard-coded to 0 in the controller; the model expects exactly one of them to be 1.
# DDOS_ML_DETECTION
