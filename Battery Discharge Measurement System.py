# Battery Discharge Measurement System
# newest update: 2025-01-14
# Author : Senuda Ranawaka

import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import nidaqmx
from nidaqmx.constants import LineGrouping
import time
import pandas as pd
import threading
import os
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ------------ CONFIGURATION ------------
AI_CHANNELS = "cDAQ9185-2416CF3Mod1/ai0:1"
RELAY_CHANNEL = "cDAQ9185-2416CF3Mod2/port0/line0"

CUTOFF_VOLTAGE = 2.0
SAMPLING_RATE = 10
UPDATE_INTERVAL = int(1000 / SAMPLING_RATE)

CSV_PATH = os.path.join(os.getcwd(), "battery_discharge_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "battery_time_model.pkl")

R_SHUNT = 0.1 
RATED_CAPACITY_MAH = 2600.0

# ------------ SHARED STATE (FASTAPI) ------------
shared_state = {
    "voltage": 0.0,
    "current": 0.0,
    "elapsed_time": 0,
    "remaining_time": 0,
    "graph": []
}

# ------------ GLOBAL VARIABLES ------------
running = False
start_time = None
timestamps, voltages, currents = [], [], []

ai_task = None
relay_task = None
model = None

# ------------ TKINTER WINDOW ------------
window = tk.Tk()
window.geometry("900x600")
window.title("Battery Discharge HMI")
window.config(bg="#a3a2a0")

# ------------ LOAD ML MODEL ------------
def load_ml_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        label_prediction.config(text="Estimated Remaining Time: Ready")
        print("ML model loaded")
    except Exception as e:
        print("ML load error:", e)
        label_prediction.config(text="Estimated Remaining Time: Model error")

# ------------ FEATURE ENGINEERING ------------
def compute_dV_dt():
    if len(voltages) < 5:
        return 0.0
    dv = voltages[-1] - voltages[-5]
    dt = timestamps[-1] - timestamps[-5]
    return dv / dt if dt else 0.0

# ------------ CAPACITY & HEALTH ------------
def compute_capacity_mAh(t, i):
    if len(t) < 2:
        return 0.0
    cap = 0.0
    for k in range(1, len(t)):
        cap += ((i[k] + i[k - 1]) / 2) * ((t[k] - t[k - 1]) / 3600)
    return cap * 1000

def save_data_and_health():
    if not timestamps:
        return

    df = pd.DataFrame({
        "Time_s": timestamps,
        "Voltage_V": voltages,
        "Current_A": currents
    })
    df.to_csv(CSV_PATH, index=False)
    print("Data saved to:", CSV_PATH)

    capacity = compute_capacity_mAh(timestamps, currents)
    health = 100.0 * capacity / RATED_CAPACITY_MAH if RATED_CAPACITY_MAH else 0.0

    print(f"Capacity: {capacity:.1f} mAh | Health: {health:.1f}%")

    window.after(
        0,
        lambda: (
            label_capacity.config(text=f"Capacity: {capacity:.1f} mAh"),
            label_health.config(text=f"Health: {health:.1f} %"),
            label_status.config(text="Completed")
        )
    )

# ------------ DAQ LOOP ------------
def daq_loop():
    global running, start_time, ai_task, relay_task

    ai_task = nidaqmx.Task()
    ai_task.ai_channels.add_ai_voltage_chan(AI_CHANNELS)

    relay_task = nidaqmx.Task()
    relay_task.do_channels.add_do_chan(
        RELAY_CHANNEL,
        line_grouping=LineGrouping.CHAN_PER_LINE
    )

    relay_task.write(True)
    start_time = time.time()

    try:
        while running:
            v = ai_task.read()
            V_batt, V_shunt = float(v[0]), float(v[1])
            I = V_shunt / R_SHUNT
            elapsed = time.time() - start_time

            timestamps.append(elapsed)
            voltages.append(V_batt)
            currents.append(I)

            remaining = 0
            if model and len(voltages) > 20:
                X = np.array([[V_batt, elapsed, compute_dV_dt()]])
                total = float(model.predict(X)[0])
                remaining = max(0, int(total - elapsed))

                window.after(
                    0,
                    lambda r=remaining: label_prediction.config(
                        text=f"Estimated Remaining Time: {r//60} min {r%60} s"
                    )
                )

            shared_state.update({
                "voltage": V_batt,
                "current": I,
                "elapsed_time": int(elapsed),
                "remaining_time": remaining
            })

            shared_state["graph"].append({"x": elapsed, "y": V_batt})
            if len(shared_state["graph"]) > 300:
                shared_state["graph"].pop(0)

            if V_batt <= CUTOFF_VOLTAGE:
                running = False
                break

            time.sleep(1 / SAMPLING_RATE)

    finally:
        relay_task.write(False)
        ai_task.close()
        relay_task.close()
        save_data_and_health()

# ------------ CONTROL ------------
def start():
    global running
    if running:
        return
    running = True
    timestamps.clear()
    voltages.clear()
    currents.clear()
    shared_state["graph"].clear()

    label_status.config(text="Measuring...")
    label_capacity.config(text="Capacity: -- mAh")
    label_health.config(text="Health: -- %")

    threading.Thread(target=daq_loop, daemon=True).start()
    update_plot()

def stop():
    global running
    running = False
    label_status.config(text="Stopped")

def update_plot():
    if running:
        line.set_data(timestamps, voltages)
        ax.relim()
        ax.autoscale_view()
        canvas.draw_idle()
        window.after(UPDATE_INTERVAL, update_plot)

# ------------ FASTAPI ------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
def status():
    return shared_state

@app.post("/start")
def api_start():
    start()
    return {"status": "started"}

@app.post("/stop")
def api_stop():
    stop()
    return {"status": "stopped"}

threading.Thread(
    target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000),
    daemon=True
).start()

# ------------ Tkinter HMI ------------
title = tk.Label(window, text="Battery Discharge Measurement System",
                 font=("Arial", 16, "bold"), bg="#a3a2a0")
title.pack(pady=10)

control_frame = tk.Frame(window, bg="#a3a2a0")
control_frame.place(x=20, y=100)

tk.Button(control_frame, text="START", font=("Arial", 14),
          bg="#35dc48", width=15, height=2, command=start).pack(pady=5)

tk.Button(control_frame, text="STOP", font=("Arial", 14),
          bg="#ff4c4c", width=15, height=2, command=stop).pack(pady=5)

label_status = tk.Label(window, text="Idle", font=("Arial", 12), bg="#a3a2a0")
label_status.place(x=20, y=60)

label_prediction = tk.Label(window, text="Estimated Remaining Time: --",
                            font=("Arial", 12, "bold"), fg="blue", bg="#a3a2a0")
label_prediction.place(x=250, y=60)

label_capacity = tk.Label(window, text="Capacity: -- mAh",
                          font=("Arial", 12, "bold"), bg="#a3a2a0")
label_capacity.place(x=250, y=520)

label_health = tk.Label(window, text="Health: -- %",
                        font=("Arial", 12, "bold"), bg="#a3a2a0")
label_health.place(x=500, y=520)

plot_frame = tk.Frame(window, bg="#a3a2a0")
plot_frame.place(x=250, y=90)

fig, ax = plt.subplots(figsize=(6.5, 4))
fig.patch.set_facecolor("#a3a2a0")
line, = ax.plot([], [], linewidth=2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
ax.grid(True)

canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack()

window.after(100, load_ml_model)
window.mainloop()
