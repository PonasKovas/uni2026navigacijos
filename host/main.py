import struct
import sys
import threading
import time
import zlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection
import numpy as np
import serial

PORT = "/dev/ttyACM0"
BAUDRATE = 921600
REQ_BYTE = b"\x23"  # 0x23 ('#')
MAX_CAP = 5000.0

# --- Wire Format Definitions ---
MAGIC = b"NAV1"
HEADER_FMT = "<4sIQQIIHHHHHHH3h2H"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 56 bytes
ZONES_SIZE = 64 * 4 * 2                    # 512 bytes (64 zones * 4 targets * 2 B)
SAMPLE_SIZE = 12                           # 6 * i16 (ax, ay, az, gx, gy, gz)
CRC_SIZE = 4

# --- Shared State ---
data_lock = threading.Lock()
running = True

latest_zones = None
latest_accel = (0.0, 0.0, 1.0)        # (ax, ay, az) in g
latest_gyro_rates = (0.0, 0.0, 0.0)   # (gx, gy, gz) in deg/s


def serial_worker():
    """Background thread: handles UART request-response loop with framing sync & CRC32."""
    global latest_zones, latest_accel, latest_gyro_rates, running

    buf = bytearray()

    try:
        with serial.Serial(PORT, BAUDRATE, timeout=0.1) as ser:
            ser.reset_input_buffer()
            ser.write(REQ_BYTE)
            ser.flush()

            last_req_time = time.monotonic()

            while running:
                waiting = ser.in_waiting
                chunk = ser.read(waiting if waiting > 0 else 1)
                if chunk:
                    buf.extend(chunk)

                # Search for packet magic sync b"NAV1"
                idx = buf.find(MAGIC)
                if idx < 0:
                    if len(buf) > 3:
                        del buf[:-3]
                    # If timeout waiting for MCU response, re-send request
                    if time.monotonic() - last_req_time > 0.5:
                        ser.write(REQ_BYTE)
                        ser.flush()
                        last_req_time = time.monotonic()
                    continue

                if idx > 0:
                    del buf[:idx]

                # Wait for at least the full header
                if len(buf) < HEADER_SIZE:
                    continue

                # Parse header to determine variable IMU sample count
                header_bytes = bytes(buf[:HEADER_SIZE])
                (
                    magic,
                    vl53_seq,
                    vl53_t_us,
                    imu_t_us,
                    imu_seq0,
                    imu_lost,
                    n_imu,
                    flags,
                    imu_rate_hz,
                    imu_resets,
                    imu_errors,
                    accel_lsb_per_g,
                    gyro_lsb_per_dps_x10,
                    bias_x,
                    bias_y,
                    bias_z,
                    res0,
                    res1,
                ) = struct.unpack(HEADER_FMT, header_bytes)

                # Sanity check n_imu (firmware MAX_BATCH is 64)
                if n_imu > 128:
                    del buf[:4]
                    continue

                total_packet_len = HEADER_SIZE + ZONES_SIZE + n_imu * SAMPLE_SIZE + CRC_SIZE

                if len(buf) < total_packet_len:
                    if time.monotonic() - last_req_time > 1.0:
                        del buf[:4]
                        ser.write(REQ_BYTE)
                        ser.flush()
                        last_req_time = time.monotonic()
                    continue

                packet = bytes(buf[:total_packet_len])
                del buf[:total_packet_len]

                # Validate CRC-32 (IEEE / zlib)
                body = packet[:-CRC_SIZE]
                pkt_crc = struct.unpack("<I", packet[-CRC_SIZE:])[0]
                calc_crc = zlib.crc32(body) & 0xFFFFFFFF

                if calc_crc != pkt_crc:
                    print(f"[CRC Error] expected 0x{pkt_crc:08x}, got 0x{calc_crc:08x}", file=sys.stderr)
                    ser.write(REQ_BYTE)
                    ser.flush()
                    last_req_time = time.monotonic()
                    continue

                # Parse ToF zones
                raw_zones = packet[HEADER_SIZE : HEADER_SIZE + ZONES_SIZE]
                zones = np.frombuffer(raw_zones, dtype="<i2").reshape((64, 4))

                # Parse IMU samples
                raw_imu = packet[HEADER_SIZE + ZONES_SIZE : HEADER_SIZE + ZONES_SIZE + n_imu * SAMPLE_SIZE]
                if n_imu > 0:
                    imu_samples = np.frombuffer(raw_imu, dtype="<i2").reshape((n_imu, 6))
                else:
                    imu_samples = np.zeros((0, 6), dtype=np.int16)

                # Scale factors from header
                acc_scale = float(accel_lsb_per_g) if accel_lsb_per_g > 0 else 16384.0
                gyro_scale = (float(gyro_lsb_per_dps_x10) / 10.0) if gyro_lsb_per_dps_x10 > 0 else 65.5

                with data_lock:
                    latest_zones = zones

                    if n_imu > 0:
                        last_s = imu_samples[-1]
                        latest_accel = (
                            float(last_s[0]) / acc_scale,
                            float(last_s[1]) / acc_scale,
                            float(last_s[2]) / acc_scale,
                        )

                        cur_gx = (float(last_s[3]) - bias_x) / gyro_scale
                        cur_gy = (float(last_s[4]) - bias_y) / gyro_scale
                        cur_gz = (float(last_s[5]) - bias_z) / gyro_scale
                        latest_gyro_rates = (cur_gx, cur_gy, cur_gz)

                # Request next packet
                ser.write(REQ_BYTE)
                ser.flush()
                last_req_time = time.monotonic()

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

        fig.suptitle(
            "VL53L5CX ToF & MPU-6050 IMU | Low = Bright | Negative = Gray | Press 'q' to quit",
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