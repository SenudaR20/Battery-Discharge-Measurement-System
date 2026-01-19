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

# ------------ CONFIGURATION ------------
# AI0: Battery voltage
# AI1: Shunt voltage (for current measurement)
AI_CHANNELS = "cDAQ9185-2416CF3Mod1/ai0:1"   # ai0 and ai1
RELAY_CHANNEL = "cDAQ9185-2416CF3Mod2/port0/line0"

CUTOFF_VOLTAGE = 2.0           # Pack cutoff voltage (2 NiMH in series)
SAMPLING_RATE = 10             # Hz
UPDATE_INTERVAL = int(1000 / SAMPLING_RATE)

CSV_PATH = os.path.join(os.getcwd(), "battery_discharge_data.csv")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "battery_time_model.pkl"   # model should predict TOTAL discharge time (s)
)

R_SHUNT = 0.1                  # Ohms (shunt resistor)
RATED_CAPACITY_MAH = 2000.0    # Rated capacity for 2-cell pack

# ------------ GLOBAL VARIABLES ------------
running = False
start_time = None
timestamps = []
voltages = []
currents = []   # measured via shunt

ai_task = None
relay_task = None
daq_thread = None
model = None

# ------------ TKINTER WINDOW ------------
window = tk.Tk()
window.geometry("900x600")
window.title("Battery Discharge HMI")
window.config(bg="#a3a2a0")

# ------------ LOAD ML MODEL (SAFE) ------------
def load_ml_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print("ML model loaded successfully from:", MODEL_PATH)
        label_prediction.config(text="Estimated Remaining Time: Ready")
    except Exception as e:
        print("ERROR loading ML model:", e)
        label_prediction.config(text="Estimated Remaining Time: Model error")

# ------------ ML FEATURE ENGINEERING ------------
def compute_dV_dt():
    if len(voltages) < 5:
        return 0.0
    dv = voltages[-1] - voltages[-5]
    dt = timestamps[-1] - timestamps[-5]
    return dv / dt if dt != 0 else 0.0

# ------------ CAPACITY CALCULATION ------------
def compute_capacity_mAh(timestamps_list, currents_list):
    """
    Integrate current over time using trapezoidal rule.
    timestamps: seconds
    currents: amps
    Returns capacity in mAh.
    """
    if len(timestamps_list) < 2:
        return 0.0

    capacity_Ah = 0.0
    for i in range(1, len(timestamps_list)):
        dt = timestamps_list[i] - timestamps_list[i - 1]  # seconds
        I_avg = (currents_list[i] + currents_list[i - 1]) / 2.0
        capacity_Ah += I_avg * (dt / 3600.0)  # convert to hours

    return capacity_Ah * 1000.0  # mAh

# ------------ DAQ THREAD ------------
def daq_loop():
    global running, start_time, ai_task, relay_task

    ai_task = nidaqmx.Task()
    # RSE mode: ai0 and ai1 referenced to COM of module
    ai_task.ai_channels.add_ai_voltage_chan(AI_CHANNELS)

    relay_task = nidaqmx.Task()
    relay_task.do_channels.add_do_chan(
        RELAY_CHANNEL,
        line_grouping=LineGrouping.CHAN_PER_LINE
    )

    relay_task.write(True)
    print("Relay ON → Battery connected")

    start_time = time.time()

    try:
        while running:
            # Read both channels: [V_batt, V_shunt]
            v_read = ai_task.read()
            # If nidaqmx returns a scalar when single channel, but list for multi:
            if isinstance(v_read, list) or isinstance(v_read, tuple):
                V_batt = float(v_read[0])
                V_shunt = float(v_read[1])
            else:
                # Fallback (should not happen with ai0:1)
                V_batt = float(v_read)
                V_shunt = 0.0

            I = V_shunt / R_SHUNT   # current in A

            elapsed = time.time() - start_time

            timestamps.append(elapsed)
            voltages.append(V_batt)
            currents.append(I)

            # ------------ ML PREDICTION ------------
            # Only predict if model loaded and we have enough data
            if model is not None and len(voltages) > 20:
                dV_dt = compute_dV_dt()
                # Feature vector must match what you used in training
                X_live = np.array([[V_batt, elapsed, dV_dt]])

                # Model predicts TOTAL discharge time (seconds)
                total_time_pred = float(model.predict(X_live)[0])

                # Remaining time = predicted total - elapsed
                remaining_time = max(0.0, total_time_pred - elapsed)

                minutes = int(remaining_time // 60)
                seconds = int(remaining_time % 60)

                window.after(
                    0,
                    lambda m=minutes, s=seconds: label_prediction.config(
                        text=f"Estimated Remaining Time: {m} min {s} s"
                    )
                )
            else:
                # Not enough data yet or no model
                window.after(
                    0,
                    lambda: label_prediction.config(
                        text="Estimated Remaining Time: Collecting data..."
                    )
                )

            print(
                f"Time: {elapsed:.1f}s | "
                f"V_batt: {V_batt:.3f}V | "
                f"I: {I:.3f}A"
            )

            if V_batt <= CUTOFF_VOLTAGE:
                print("Cutoff voltage reached")
                running = False
                break

            time.sleep(1 / SAMPLING_RATE)

    finally:
        # Turn relay off, close tasks, save data, compute capacity/health
        try:
            relay_task.write(False)
        except Exception:
            pass
        relay_task.close()
        ai_task.close()
        save_data_and_health()
        print("Relay OFF → Battery disconnected")

# ------------ CONTROL FUNCTIONS ------------
def start():
    global running, daq_thread

    if running:
        return

    running = True
    timestamps.clear()
    voltages.clear()
    currents.clear()

    label_status.config(text="Measuring...")
    label_prediction.config(text="Estimated Remaining Time: Calculating...")
    label_capacity.config(text="Capacity: -- mAh")
    label_health.config(text="Health: -- %")

    daq_thread = threading.Thread(target=daq_loop, daemon=True)
    daq_thread.start()

    update_plot()

def stop():
    global running
    running = False
    label_status.config(text="Stopped")
    label_prediction.config(text="Estimated Remaining Time: --")
    print("Measurement stopped")

def update_plot():
    if not running:
        return

    line.set_data(timestamps, voltages)
    ax.relim()
    ax.autoscale_view()

    canvas.draw_idle()
    window.after(UPDATE_INTERVAL, update_plot)

def save_data_and_health():
    if not timestamps:
        return

    # Save CSV with time, voltage, and current
    df = pd.DataFrame({
        "Time_s": timestamps,
        "Voltage_V": voltages,
        "Current_A": currents
    })
    df.to_csv(CSV_PATH, index=False)
    print("Data saved to:", CSV_PATH)

    # Compute capacity and health
    capacity_mAh = compute_capacity_mAh(timestamps, currents)
    health_pct = 0.0
    if RATED_CAPACITY_MAH > 0:
        health_pct = 100.0 * capacity_mAh / RATED_CAPACITY_MAH

    print(f"Measured Capacity: {capacity_mAh:.1f} mAh")
    print(f"Estimated Health: {health_pct:.1f} %")

    # Update GUI labels
    window.after(
        0,
        lambda cap=capacity_mAh, hp=health_pct: (
            label_capacity.config(text=f"Capacity: {cap:.1f} mAh"),
            label_health.config(text=f"Health: {hp:.1f} %"),
            label_status.config(text="Completed")
        )
    )

# ------------ UI ELEMENTS ------------
title = tk.Label(
    window,
    text="Battery Discharge Measurement System",
    font=("Arial", 16, "bold"),
    bg="#a3a2a0"
)
title.pack(pady=10)

control_frame = tk.Frame(window, bg="#a3a2a0")
control_frame.place(x=20, y=100)

tk.Button(
    control_frame,
    text="START",
    font=("Arial", 14),
    bg="#35dc48",
    width=15,
    height=2,
    command=start
).pack(pady=5)

tk.Button(
    control_frame,
    text="STOP",
    font=("Arial", 14),
    bg="#ff4c4c",
    width=15,
    height=2,
    command=stop
).pack(pady=5)

label_status = tk.Label(
    window,
    text="Idle",
    font=("Arial", 12),
    bg="#a3a2a0"
)
label_status.place(x=20, y=60)

label_prediction = tk.Label(
    window,
    text="Estimated Remaining Time: Loading model...",
    font=("Arial", 12, "bold"),
    fg="blue",
    bg="#a3a2a0"
)
label_prediction.place(x=250, y=60)

label_capacity = tk.Label(
    window,
    text="Capacity: -- mAh",
    font=("Arial", 12, "bold"),
    fg="black",
    bg="#a3a2a0"
)
label_capacity.place(x=250, y=520)

label_health = tk.Label(
    window,
    text="Health: -- %",
    font=("Arial", 12, "bold"),
    fg="black",
    bg="#a3a2a0"
)
label_health.place(x=500, y=520)

# ------------ PLOT AREA ------------
plot_frame = tk.Frame(window, bg="#a3a2a0")
plot_frame.place(x=250, y=90)

fig, ax = plt.subplots(figsize=(6.5, 4))
fig.patch.set_facecolor("#a3a2a0")

line, = ax.plot([], [], color="blue", linewidth=2)

ax.set_title("Battery Discharge Curve")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
ax.grid(True)

canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack()

# ------------ SAFE EXIT ------------
def on_close():
    global running
    running = False
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_close)

# ------------ START ML LOAD AFTER GUI ------------
window.after(100, load_ml_model)

# ------------ MAIN LOOP ------------
window.mainloop()