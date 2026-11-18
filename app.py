#!/usr/bin/env python3
"""
DDoS detection & mitigation demo - web front end + simulation server.

Run:   python3 app.py
Open:  http://127.0.0.1:5000

The model is loaded from ddos_model.pkl (same folder by default).
Override with:  DDOS_MODEL=/path/to/model.pkl python3 app.py
"""
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from sim import Simulation, Runner

HERE = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("DDOS_MODEL", HERE / "ddos_model.pkl"))

app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")
sim = Simulation(MODEL_PATH)
Runner(sim).start()


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/state")
def state():
    return jsonify(sim.snapshot())


@app.post("/api/attack")
def attack():
    body = request.get_json(silent=True) or {}
    if body.get("action") == "stop":
        sim.stop_attack()
    else:
        sim.start_attack(body.get("intensity", "medium"))
    return jsonify(ok=True)


@app.post("/api/mitigation")
def mitigation():
    body = request.get_json(silent=True) or {}
    sim.set_auto_mitigation(bool(body.get("on", True)))
    return jsonify(ok=True)


@app.post("/api/reset")
def reset():
    sim.reset()
    return jsonify(ok=True)


if __name__ == "__main__":
    print(f"Model path: {MODEL_PATH}  (loaded: {sim.detector.loaded})")
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), threaded=True)
