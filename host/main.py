import sys
import threading
import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import serial

PORT = "/dev/ttyACM0"
BAUDRATE = 921600
PAYLOAD_SIZE = 512  # 64 samples * 4 channels * 2 bytes (i16) = 512 bytes
REQ_BYTE = b"\x23"  # 0x23 ('#')
MAX_CAP = 5000.0

# --- Shared State ---
latest_data = None
data_lock = threading.Lock()
running = True


def read_exact(ser: serial.Serial, num_bytes: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < num_bytes and running:
        chunk = ser.read(num_bytes - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def serial_worker():
    """Background thread: handles UART request-response loop at full speed."""
    global latest_data, running

    try:
        with serial.Serial(PORT, BAUDRATE, timeout=1.0) as ser:
            ser.reset_input_buffer()
            ser.write(REQ_BYTE)
            ser.flush()

            while running:
                raw = read_exact(ser, PAYLOAD_SIZE)
                if raw is None or len(raw) < PAYLOAD_SIZE:
                    if running:
                        ser.write(REQ_BYTE)
                        ser.flush()
                    continue

                # Parse into shape (64, 4) of little-endian int16
                arr = np.frombuffer(raw, dtype="<i2").reshape((64, 4))

                with data_lock:
                    latest_data = arr

                # Signal MCU for next payload
                ser.write(REQ_BYTE)
                ser.flush()

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
            if latest_data is not None:
                break
        time.sleep(0.02)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Inverted colormap (low values = bright yellow, high = dark purple)
    # .set_bad renders negative/masked values as gray
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(color="gray")

    dummy_grid = np.ma.masked_less(np.zeros((8, 8)), 0)
    images = []
    cbars = []

    for i, ax in enumerate(axes):
        im = ax.imshow(dummy_grid, cmap=cmap, interpolation="nearest", vmin=0, vmax=MAX_CAP)
        ax.set_title(f"Channel {i}")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        images.append(im)
        cbars.append(cbar)

    plt.tight_layout()

    def update(_):
        with data_lock:
            if latest_data is None:
                return images
            data = latest_data.copy()

        fig.suptitle(
            "Momentary Limits (Max ≤ 5000) | Low = Bright | Negative = Gray | Press 'q' to quit",
            fontsize=12,
            fontweight="bold",
        )

        for ch in range(4):
            raw_grid = data[:, ch].reshape((8, 8))

            # Mask out any value < 0 (drawn as gray)
            masked_grid = np.ma.masked_less(raw_grid, 0)
            images[ch].set_data(masked_grid)

            # Momentary limits: compute strictly from current frame's positive values
            valid = raw_grid[raw_grid >= 0]
            if valid.size > 0:
                vmin = max(0.0, float(valid.min()))
                vmax = min(MAX_CAP, float(valid.max()))
            else:
                vmin = 0.0
                vmax = MAX_CAP

            # Prevent zero or inverted range crashes
            if vmin >= vmax:
                if vmax >= MAX_CAP:
                    vmin = max(0.0, MAX_CAP - 1.0)
                    vmax = MAX_CAP
                else:
                    vmax = vmin + 1.0

            images[ch].set_clim(vmin, vmax)
            cbars[ch].update_normal(images[ch])

        return images

    ani = animation.FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)

    try:
        plt.show()
    finally:
        running = False
        print("\nExiting...")


if __name__ == "__main__":
    main()