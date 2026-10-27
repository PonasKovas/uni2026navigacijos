import sys
import threading
import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection
import numpy as np
import serial

import navlink

MAX_CAP = 5000.0

# --- Shared State ---
data_lock = threading.Lock()
running = True

latest_zones = None
latest_accel = (0.0, 0.0, 1.0)        # (ax, ay, az) in g
latest_gyro_rates = (0.0, 0.0, 0.0)   # (gx, gy, gz) in deg/s
latest_temp = None                    # MPU die temperature in °C


def serial_worker():
    """Background thread: keeps the latest packet values for the plot."""
    global latest_zones, latest_accel, latest_gyro_rates, latest_temp

    try:
        with navlink.open_port() as ser:
            for pkt in navlink.read_packets(ser):
                if not running:
                    break
                with data_lock:
                    latest_zones = pkt.zones
                    if pkt.n_imu > 0:
                        latest_accel = tuple(float(v) for v in pkt.accel_g[-1])
                        latest_gyro_rates = tuple(float(v) for v in pkt.gyro_dps[-1])
                        # average the batch: one sample's temp is only ~0.003 °C/LSB but noisy
                        latest_temp = float(pkt.temp_c.mean())

    except serial.SerialException as e:
        print(f"\n[Serial Error] {e}", file=sys.stderr)
    except Exception as e:
        print(f"\n[Worker Error] {e}", file=sys.stderr)


def on_key(event):
    if event.key in ("q", "escape"):
        plt.close("all")


def main():
    global running

    # Start serial background thread
    t = threading.Thread(target=serial_worker, daemon=True)
    t.start()

    # Wait briefly for first packet
    for _ in range(50):
        with data_lock:
            if latest_zones is not None:
                break
        time.sleep(0.02)

    # Setup layout: 2 rows (Row 0: 4 ToF heatmaps, Row 1: Accel 3D + Gyro bars)
    fig = plt.figure(figsize=(16, 9))
    fig.canvas.mpl_connect("key_press_event", on_key)

    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.3)

    # --- Top Row: 4 ToF Heatmaps ---
    axes_tof = [fig.add_subplot(gs[0, i]) for i in range(4)]
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(color="gray")

    dummy_grid = np.ma.masked_less(np.zeros((8, 8)), 0)
    images = []
    cbars = []

    for i, ax in enumerate(axes_tof):
        im = ax.imshow(dummy_grid, cmap=cmap, interpolation="nearest", vmin=0, vmax=MAX_CAP)
        ax.set_title(f"Channel {i}")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        images.append(im)
        cbars.append(cbar)

    # --- Bottom Left: 3D Accelerometer Vector ---
    ax_acc = fig.add_subplot(gs[1, :2], projection="3d")
    ax_acc.set_xlim([-1.5, 1.5])
    ax_acc.set_ylim([-1.5, 1.5])
    ax_acc.set_zlim([-1.5, 1.5])
    ax_acc.set_xlabel("X (g)")
    ax_acc.set_ylabel("Y (g)")
    ax_acc.set_zlabel("Z (g)")
    ax_acc.set_box_aspect([1, 1, 1])

    (acc_line,) = ax_acc.plot([0, 0], [0, 0], [0, 0], color="#e74c3c", linewidth=3.5)
    (acc_tip,) = ax_acc.plot([0], [0], [0], marker="o", markersize=8, color="#c0392b")
    (acc_proj,) = ax_acc.plot([0, 0], [0, 0], [0, 0], color="gray", linestyle=":", linewidth=1.2)
    ax_acc.plot([0], [0], [0], marker="o", markersize=4, color="black")

    # --- Bottom Right: Gyroscope Bidirectional Bar Chart ---
    ax_gyro = fig.add_subplot(gs[1, 2:])
    ax_gyro.set_ylabel("Angular Velocity (°/s)")
    ax_gyro.set_ylim(-550, 550)
    ax_gyro.set_yticks([-500, -250, 0, 250, 500])
    ax_gyro.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_gyro.grid(True, linestyle=":", alpha=0.5)

    axis_names = ["X", "Y", "Z"]
    bar_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    bars = ax_gyro.bar(axis_names, [0.0, 0.0, 0.0], color=bar_colors, width=0.45)

    bar_texts = [
        ax_gyro.text(i, 20, "0.0 °/s", ha="center", va="bottom", fontsize=10, fontweight="bold")
        for i in range(3)
    ]

    def update(_):
        with data_lock:
            if latest_zones is None:
                return images
            zones = latest_zones.copy()
            acc = latest_accel
            rates = latest_gyro_rates
            temp = latest_temp

        temp_str = f"{temp:.2f} °C" if temp is not None else "-- °C"
        fig.suptitle(
            f"VL53L5CX ToF & MPU-6050 IMU ({temp_str}) | Low = Bright | Negative = Gray | Press 'q' to quit",
            fontsize=12,
            fontweight="bold",
        )

        # 1. Update ToF Channel Grids
        for ch in range(4):
            raw_grid = zones[:, ch].reshape((8, 8))
            masked_grid = np.ma.masked_less(raw_grid, 0)
            images[ch].set_data(masked_grid)

            valid = raw_grid[raw_grid >= 0]
            if valid.size > 0:
                vmin = max(0.0, float(valid.min()))
                vmax = min(MAX_CAP, float(valid.max()))
            else:
                vmin = 0.0
                vmax = MAX_CAP

            if vmin >= vmax:
                if vmax >= MAX_CAP:
                    vmin = max(0.0, MAX_CAP - 1.0)
                    vmax = MAX_CAP
                else:
                    vmax = vmin + 1.0

            images[ch].set_clim(vmin, vmax)
            cbars[ch].update_normal(images[ch])

        # 2. Update 3D Accelerometer Vector
        ax_val, ay_val, az_val = acc
        norm_a = float(np.sqrt(ax_val**2 + ay_val**2 + az_val**2))
        acc_line.set_data([0, ax_val], [0, ay_val])
        acc_line.set_3d_properties([0, az_val])
        acc_tip.set_data([ax_val], [ay_val])
        acc_tip.set_3d_properties([az_val])
        acc_proj.set_data([ax_val, ax_val], [ay_val, ay_val])
        acc_proj.set_3d_properties([0, az_val])
        ax_acc.set_title(
            f"Accelerometer 3D Vector: |a| = {norm_a:.2f} g\n(X: {ax_val:+.2f}, Y: {ay_val:+.2f}, Z: {az_val:+.2f})",
            fontsize=10,
            fontweight="bold",
        )

        # 3. Update Gyroscope Bidirectional Bar Chart
        for i, (bar, txt, rate) in enumerate(zip(bars, bar_texts, rates)):
            if rate >= 0:
                bar.set_y(0)
                bar.set_height(rate)
                txt.set_position((i, rate + 15))
                txt.set_va("bottom")
            else:
                bar.set_y(rate)
                bar.set_height(-rate)
                txt.set_position((i, rate - 15))
                txt.set_va("top")
            txt.set_text(f"{rate:+.1f} °/s")

        ax_gyro.set_title(
            f"Gyroscope: X: {rates[0]:+.1f} °/s | Y: {rates[1]:+.1f} °/s | Z: {rates[2]:+.1f} °/s",
            fontsize=10,
            fontweight="bold",
        )

        return images

    ani = animation.FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)

    try:
        plt.show()
    finally:
        running = False
        print("\nExiting...")


if __name__ == "__main__":
    main()