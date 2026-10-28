"""IMU accuracy experiment: record raw IMU data to CSV, then analyze it offline.

    python imu_experiment.py record run1.csv            # Enter = mark (type a label first if you want)
    python imu_experiment.py record still.csv -d 1800   # stop by itself after 30 min
    python imu_experiment.py analyze run1.csv           # rotate-and-hold runs: still segments + angles
    python imu_experiment.py stationary still.csv       # long still logs: Allan deviation + drift vs temp
    python imu_experiment.py calib6 calib.csv           # hold ±X ±Y ±Z up ~10 s each: accel calibration

Log format: comment lines start with '#'. "# key=value" is metadata, "# mark,<n>,<host_t>,<label>"
is a mark. Every other line is one IMU sample with raw LSB values (gyro NOT bias corrected):
    seq, host_t, mcu_t_us, flags, ax, ay, az, gx, gy, gz, temp
host_t (s since record start) and mcu_t_us are per packet, so ~13 samples share them.
"""

import argparse
import datetime
import queue
import sys
import threading
import time

import numpy as np

import navlink

COLUMNS = "seq,host_t,mcu_t_us,flags,ax,ay,az,gx,gy,gz,temp"
STATUS_PERIOD_S = 0.5
MAX_TRUSTED_GAP_S = 0.2  # interpolating across longer gaps is mostly guessing


# ============================================================ record

def _stdin_marks(q):
    """Each line typed on stdin becomes a mark, timed when Enter is pressed."""
    for line in sys.stdin:
        q.put((time.monotonic(), line.strip()))


def cmd_record(args):
    marks = queue.Queue()
    threading.Thread(target=_stdin_marks, args=(marks,), daemon=True).start()

    n_samples = n_packets = n_gaps = n_clipped = n_marks = 0
    next_seq = None
    angle = np.zeros(3)  # rough live angle, boot bias only
    temp = float("nan")
    last_status = 0.0

    with navlink.open_port(args.port) as ser, open(args.out, "w") as f:
        t_ref = time.monotonic()
        print("recording, Ctrl+C to stop. Enter = mark (type a label before Enter if you want)",
              file=sys.stderr)
        try:
            for pkt in navlink.read_packets(ser):
                if n_packets == 0:
                    f.write("# nav-imu-log v1\n")
                    f.write(f"# start_wall={datetime.datetime.now().isoformat(timespec='seconds')}\n")
                    f.write(f"# imu_rate_hz={pkt.imu_rate_hz}\n")
                    f.write(f"# accel_lsb_per_g={pkt.accel_scale}\n")
                    f.write(f"# gyro_lsb_per_dps={pkt.gyro_scale}\n")
                    f.write(f"# boot_gyro_bias={','.join(str(b) for b in pkt.gyro_bias)}\n")
                    f.write(f"# boot_bias_valid={int(bool(pkt.flags & navlink.FLAG_IMU_BIAS_VALID))}\n")
                    f.write(f"# columns={COLUMNS}\n")
                n_packets += 1

                host_t = pkt.host_t - t_ref
                # the first packet's gap flag only means nobody was listening before we connected
                if next_seq is not None and (
                    pkt.flags & navlink.FLAG_IMU_GAP_BEFORE or (pkt.n_imu > 0 and pkt.imu_seq0 != next_seq)
                ):
                    n_gaps += 1
                if pkt.flags & navlink.FLAG_IMU_CLIPPED:
                    n_clipped += 1

                for i, s in enumerate(pkt.imu):
                    f.write(f"{pkt.imu_seq0 + i},{host_t:.6f},{pkt.imu_t_us},{pkt.flags},"
                            + ",".join(str(int(v)) for v in s) + "\n")
                if pkt.n_imu > 0:
                    next_seq = pkt.imu_seq0 + pkt.n_imu
                    n_samples += pkt.n_imu
                    angle += pkt.gyro_dps.sum(axis=0) / pkt.imu_rate_hz
                    temp = float(pkt.temp_c.mean())

                while not marks.empty():
                    mark_t, label = marks.get()
                    n_marks += 1
                    f.write(f"# mark,{n_marks},{mark_t - t_ref:.6f},{label}\n")
                    print(f"\nmark {n_marks} at {mark_t - t_ref:.2f} s {label}", file=sys.stderr)

                now = time.monotonic()
                if now - last_status > STATUS_PERIOD_S:
                    last_status = now
                    sys.stderr.write(
                        f"\r{host_t:7.1f} s  {n_samples:7d} samples  gaps {n_gaps}  clipped {n_clipped}"
                        f"  {temp:6.2f} °C  angle≈ X{angle[0]:+8.2f} Y{angle[1]:+8.2f} Z{angle[2]:+8.2f}°  "
                    )
                    sys.stderr.flush()

                if args.duration and host_t >= args.duration:
                    break
        except KeyboardInterrupt:
            pass

    print(f"\nsaved {n_samples} samples ({n_gaps} gaps, {n_marks} marks) to {args.out}", file=sys.stderr)


# ============================================================ loading

class Log:
    def __init__(self, path, rate_source="host", until=None):
        self.meta = {}
        self.marks = []  # (n, host_t, label)
        with open(path) as f:
            for line in f:
                if line.startswith("# mark,"):
                    _, n, t, label = line.rstrip("\n").split(",", 3)
                    self.marks.append((int(n), float(t), label))
                elif line.startswith("# ") and "=" in line:
                    k, v = line[2:].rstrip("\n").split("=", 1)
                    self.meta[k] = v

        d = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
        if until:
            d = d[d[:, 1] <= until]
        if len(d) == 0:
            sys.exit(f"{path}: no samples")
        self.seq = d[:, 0].astype(np.int64)
        self.host_t = d[:, 1]
        self.mcu_t = d[:, 2] * 1e-6
        self.flags = d[:, 3].astype(int)

        acc_scale = float(self.meta.get("accel_lsb_per_g", 16384))
        gyro_scale = float(self.meta.get("gyro_lsb_per_dps", 65.5))
        self.acc = d[:, 4:7] / acc_scale              # g
        self.gyro = d[:, 7:10] / gyro_scale           # °/s, raw (no bias removed)
        self.temp = navlink.temp_to_c(d[:, 10])       # °C
        self.boot_bias = np.array([float(b) for b in self.meta["boot_gyro_bias"].split(",")]) / gyro_scale

        self.n_clipped = int(np.count_nonzero(self.flags & navlink.FLAG_IMU_CLIPPED))
        nominal = float(self.meta.get("imu_rate_hz", 200))
        last = np.r_[np.diff(self.host_t) != 0, True]  # last sample of each packet
        first = np.r_[True, last[:-1]]                  # first sample of each packet
        have_mcu = np.ptp(self.mcu_t[last]) > 0

        # Gaps. A gap shows as a seq jump (samples dropped on the MCU or lost on the UART) or
        # as the gap flag on a packet (FIFO reset, seq keeps counting). Row 0's flag only
        # means nobody was listening before the recording started.
        flagged = first & (self.flags & navlink.FLAG_IMU_GAP_BEFORE != 0)
        flagged[0] = False
        gap_rows = np.flatnonzero(flagged | np.r_[False, np.diff(self.seq) != 1])

        # How many samples each gap is missing. A seq jump says it exactly; a FIFO reset
        # doesn't, so measure the gap on the MCU clock (packet times are per last sample).
        rate_mcu = nominal
        if have_mcu and last.sum() >= 10:
            rate_mcu = np.polyfit(self.mcu_t[last], self.seq[last], 1)[0]
        pkt_end = np.flatnonzero(last)
        missing = np.zeros(len(self.seq), dtype=np.int64)  # missing before row i
        for i in gap_rows:
            n = int(self.seq[i] - self.seq[i - 1] - 1)
            if have_mcu:
                c = pkt_end[np.searchsorted(pkt_end, i)]  # last row of the packet holding i
                t_i = self.mcu_t[c] - (c - i) / rate_mcu
                n = max(n, int(round((t_i - self.mcu_t[i - 1]) * rate_mcu)) - 1)
            missing[i] = max(n, 0)
        pos = np.arange(len(self.seq)) + np.cumsum(missing)  # each row's index on a gap-free timeline

        # Rate estimate. The MPU's clock is not exact, so fit the sample index against a real
        # clock at the last sample of each packet. Host clock is the most accurate; USB jitter
        # averages out over a long log.
        self.rates = {"nominal": nominal}
        if last.sum() >= 10 and self.host_t[last][-1] - self.host_t[last][0] > 5:
            self.rates["host"] = np.polyfit(self.host_t[last], pos[last], 1)[0]
            if have_mcu:
                self.rates["mcu"] = np.polyfit(self.mcu_t[last], pos[last], 1)[0]
        try:
            self.rate = float(rate_source)
        except ValueError:
            self.rate = self.rates.get(rate_source, nominal)

        # Fill gaps by linear interpolation between the samples on both sides, so the
        # integration keeps the right timing and roughly the right rotation across them.
        self.gaps = [(pos[i] - missing[i], int(missing[i])) for i in gap_rows]  # (start index, length)
        if missing.any():
            full = np.arange(pos[-1] + 1)
            interp = lambda a: np.column_stack([np.interp(full, pos, a[:, k]) for k in range(a.shape[1])])
            self.acc, self.gyro = interp(self.acc), interp(self.gyro)
            self.temp = np.interp(full, pos, self.temp)
        self.filled = np.ones(pos[-1] + 1, dtype=bool)
        self.filled[pos] = False

        self.t = np.arange(pos[-1] + 1) / self.rate
        # map host time (used by marks) to sample time
        self._host_last, self._t_last = self.host_t[last], pos[last] / self.rate

    def host_to_t(self, host_t):
        return np.interp(host_t, self._host_last, self._t_last)

    def summary(self):
        print(f"{len(self.seq)} samples, {self.t[-1]:.1f} s, temp {self.temp.min():.2f}..{self.temp.max():.2f} °C")
        print("rate [Hz]: " + "  ".join(f"{k} {v:.4f}" for k, v in self.rates.items())
              + f"  -> using {self.rate:.4f}")
        if self.gaps:
            print(f"WARNING: {len(self.gaps)} gaps, {int(self.filled.sum())} samples filled by interpolation:  "
                  + ", ".join(f"{s / self.rate:.1f} s (+{n * 1000 / self.rate:.0f} ms)" for s, n in self.gaps[:10]))
            long = [(s, n) for s, n in self.gaps if n / self.rate > MAX_TRUSTED_GAP_S]
            if long:
                print(f"WARNING: {len(long)} gaps longer than {MAX_TRUSTED_GAP_S} s, any rotation during "
                      "them is guessed, angles after them are unreliable")
        if self.n_clipped:
            print(f"WARNING: {self.n_clipped} samples in clipped batches (rotation too fast?)")


def body_axes(up):
    """Pick a body frame from sensor axes so that body Z is the sensor axis closest to up.

    Keeps the Euler angles away from gimbal lock however the board is mounted (on its side,
    upside down). Returns (P, names): body = P @ sensor, names like ["-y", "x", "z"].
    """
    names = "xyz"
    zi = int(np.argmax(np.abs(up)))
    zs = 1.0 if up[zi] > 0 else -1.0
    xi = 0 if zi != 0 else 1
    P = np.zeros((3, 3))
    P[0, xi] = 1.0
    P[2, zi] = zs
    P[1] = np.cross(P[2], P[0])  # right-handed: y = z × x
    labels = []
    for row in P:
        i = int(np.flatnonzero(row)[0])
        labels.append(("-" if row[i] < 0 else "") + names[i])
    return P, labels


def accel_tilt(acc):
    """Pitch/roll in degrees from gravity (ZYX Euler convention, body Z up when level)."""
    ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
    roll = np.degrees(np.arctan2(ay, az))
    pitch = np.degrees(np.arctan2(-ax, np.hypot(ay, az)))
    return pitch, roll


def quat_from_euler(yaw, pitch, roll):
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def integrate_quat(omega, dt, q0):
    """Integrate body rates [rad/s] into orientation quaternions (w, x, y, z), body -> world.

    Each step applies the exact rotation for a constant rate over dt, so any mix of
    rotations about any axes is handled correctly.
    """
    n = len(omega)
    out = np.empty((n, 4))
    w, x, y, z = (float(v) for v in q0)
    for i in range(n):
        gx, gy, gz = omega[i]
        mag = (gx * gx + gy * gy + gz * gz) ** 0.5
        half = 0.5 * mag * dt
        if mag > 1e-12:
            s = np.sin(half) / mag
            dw, dx, dy, dz = np.cos(half), gx * s, gy * s, gz * s
            w, x, y, z = (
                w * dw - x * dx - y * dy - z * dz,
                w * dx + x * dw + y * dz - z * dy,
                w * dy - x * dz + y * dw + z * dx,
                w * dz + x * dy - y * dx + z * dw,
            )
            norm = (w * w + x * x + y * y + z * z) ** 0.5
            w, x, y, z = w / norm, x / norm, y / norm, z / norm
        out[i] = (w, x, y, z)
    return out


def quat_to_euler(q):
    """ZYX Euler angles in degrees: yaw (about vertical, unwrapped), pitch, roll."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return np.degrees(np.unwrap(yaw)), np.degrees(pitch), np.degrees(np.unwrap(roll))


def moving_mean(x, n):
    n = max(1, min(int(n), len(x)))
    k = np.ones(n) / n
    if x.ndim == 1:
        return np.convolve(x, k, mode="same")
    return np.column_stack([np.convolve(x[:, i], k, mode="same") for i in range(x.shape[1])])


def find_still(omega, rate, thresh_dps, min_s, trim_s):
    """Index ranges [start, end) where the smoothed rotation rate stays below thresh_dps."""
    mag = moving_mean(np.linalg.norm(omega, axis=1), 0.2 * rate)
    still = (mag < thresh_dps).astype(int)
    edges = np.flatnonzero(np.diff(np.r_[0, still, 0]))
    trim = int(trim_s * rate)
    segs = []
    for s, e in zip(edges[::2], edges[1::2]):
        s, e = s + trim, e - trim
        if (e - s) / rate >= min_s:
            segs.append((s, e))
    return segs


def estimate_bias(log, seconds):
    n = int(seconds * log.rate)
    if n > len(log.t) // 2:
        sys.exit(f"bias window {seconds} s is more than half the log")
    window = log.gyro[:n]
    bias = window.mean(axis=0)
    spread = np.ptp(moving_mean(window - bias, 0.2 * log.rate), axis=0)
    print(f"gyro bias from first {seconds:g} s [°/s]: " + "  ".join(f"{v:+.4f}" for v in bias)
          + "   (boot: " + "  ".join(f"{v:+.4f}" for v in log.boot_bias) + ")")
    if np.any(spread > 0.5):
        print("WARNING: sensor does not look still during the bias window "
              f"(smoothed spread {np.round(spread, 2)} °/s)")
    return bias


# ============================================================ analyze

def cmd_analyze(args):
    log = Log(args.log, args.rate, args.until)
    log.summary()
    bias_s = args.bias_seconds if args.bias_seconds > 0 else 5.0
    bias = log.boot_bias if args.bias_seconds == 0 else estimate_bias(log, args.bias_seconds)

    # Level frame from gravity at the start: world Z = up, yaw 0 = initial heading.
    up = log.acc[: int(bias_s * log.rate)].mean(axis=0)
    P, axis_names = body_axes(up)
    print(f"body frame: X = sensor {axis_names[0]}, Y = sensor {axis_names[1]}, Z (up) = sensor {axis_names[2]}")
    omega = (log.gyro - bias) @ P.T  # °/s, body frame
    acc = log.acc @ P.T
    pitch0, roll0 = accel_tilt((up @ P.T)[None, :])
    q0 = quat_from_euler(0.0, np.radians(pitch0[0]), np.radians(roll0[0]))

    q = integrate_quat(np.radians(omega), 1.0 / log.rate, q0)
    g_yaw, g_pitch, g_roll = quat_to_euler(q)
    a_pitch, a_roll = accel_tilt(acc)
    segs = find_still(omega, log.rate, args.still_thresh, args.min_still, args.trim)
    mark_t = [(n, log.host_to_t(ht), label) for n, ht, label in log.marks]

    rows = []
    for k, (s, e) in enumerate(segs):
        t0, t1 = log.t[s], log.t[e - 1]
        labels = " ".join(f"m{n}" + (f":{lab}" if lab else "") for n, mt, lab in mark_t if t0 <= mt <= t1)
        rows.append(dict(
            seg=k, t_start=t0, dur=t1 - t0,
            yaw=g_yaw[s:e].mean(), pitch=g_pitch[s:e].mean(), roll=g_roll[s:e].mean(),
            acc_pitch=a_pitch[s:e].mean(), acc_roll=a_roll[s:e].mean(),
            acc_pitch_std=a_pitch[s:e].std(), acc_roll_std=a_roll[s:e].std(),
            temp=log.temp[s:e].mean(), marks=labels,
        ))

    # Total rotation between consecutive holds, independent of how the sensor is mounted.
    # This is what the laser sees. Gyro: angle of q_prev^-1 * q. Accel: angle between gravity directions.
    prev_q = prev_g = None
    for r, (s, e) in zip(rows, segs):
        qm = q[s:e].mean(axis=0)
        qm /= np.linalg.norm(qm)
        gm = acc[s:e].mean(axis=0)
        gm /= np.linalg.norm(gm)
        r["gyro_rot"] = 0.0 if prev_q is None else float(np.degrees(2 * np.arccos(min(1.0, abs(qm @ prev_q)))))
        r["acc_tilt"] = 0.0 if prev_g is None else float(np.degrees(np.arccos(np.clip(gm @ prev_g, -1, 1))))
        prev_q, prev_g = qm, gm

    print(f"\n{len(rows)} still segments (|ω| < {args.still_thresh} °/s for ≥ {args.min_still} s)")
    print("gyro = quaternion-integrated gyro, acc = accelerometer tilt, g-a = gyro minus accel, Δ = change from previous segment")
    print("gyro rot = total rotation since previous segment (any axis, mod 360), acc tilt = change in gravity direction")
    print(" seg   t[s]   dur |    yaw    pitch    roll | acc pitch acc roll | g-a pitch g-a roll |"
          "    Δyaw  Δpitch   Δroll  Δapitch  Δaroll | gyro rot acc tilt |  temp  marks")
    step_keys = ("yaw", "pitch", "roll", "acc_pitch", "acc_roll")
    prev = None
    for r in rows:
        d = {k: (r[k] - prev[k]) if prev else 0.0 for k in step_keys}
        print(f"{r['seg']:4d} {r['t_start']:6.1f} {r['dur']:5.1f} |"
              f" {r['yaw']:+8.2f} {r['pitch']:+7.2f} {r['roll']:+7.2f} |"
              f"  {r['acc_pitch']:+7.2f}  {r['acc_roll']:+7.2f} |"
              f"  {r['pitch'] - r['acc_pitch']:+7.2f}  {r['roll'] - r['acc_roll']:+7.2f} |"
              f" {d['yaw']:+8.2f} {d['pitch']:+7.2f} {d['roll']:+7.2f} {d['acc_pitch']:+8.2f} {d['acc_roll']:+7.2f} |"
              f"  {r['gyro_rot']:7.2f}  {r['acc_tilt']:7.2f} |"
              f" {r['temp']:5.2f}  {r['marks']}")
        prev = r
    print(f"\nend of log: gyro yaw {g_yaw[-1]:+.2f}  pitch {g_pitch[-1]:+.2f}  roll {g_roll[-1]:+.2f} °")

    if rows:
        out = args.log.rsplit(".", 1)[0] + ".segments.csv"
        keys = list(rows[0].keys())
        with open(out, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
        print(f"segments saved to {out}")

    if args.no_plot:
        return
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(14, 10))
    names, colors = "XYZ", ["#e74c3c", "#2ecc71", "#3498db"]
    for i in range(3):
        axs[0].plot(log.t, omega[:, i], color=colors[i], lw=0.6, label=f"body {names[i]} (sensor {axis_names[i]})")
    axs[1].plot(log.t, g_yaw, color="#3498db", lw=1, label="gyro yaw")
    axs[2].plot(log.t, a_pitch, color="#e67e22", lw=0.5, alpha=0.5, label="acc pitch")
    axs[2].plot(log.t, a_roll, color="#8e44ad", lw=0.5, alpha=0.5, label="acc roll")
    axs[2].plot(log.t, g_pitch, color="#d35400", lw=1.5, label="gyro pitch")
    axs[2].plot(log.t, g_roll, color="#5b2c6f", lw=1.5, label="gyro roll")
    axs[3].plot(log.t, log.temp, color="black", lw=0.8)
    axs[0].set_ylabel("rate [°/s]")
    axs[1].set_ylabel("yaw [°]")
    axs[2].set_ylabel("pitch / roll [°]")
    axs[3].set_ylabel("temp [°C]")
    axs[3].set_xlabel("time [s]")
    for ax in axs:
        for s, e in segs:
            ax.axvspan(log.t[s], log.t[e - 1], color="gray", alpha=0.15, lw=0)
        for n, mt, lab in mark_t:
            ax.axvline(mt, color="black", lw=0.8, ls="--")
        ax.grid(True, ls=":", alpha=0.5)
    for n, mt, lab in mark_t:
        axs[1].annotate(f"m{n} {lab}", (mt, 1), xycoords=("data", "axes fraction"), fontsize=8,
                        rotation=90, va="top", ha="right")
    for ax in axs[:3]:
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{args.log}  (gray = still segments, dashed = marks)")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
    plt.show()


# ============================================================ stationary

def allan_deviation(omega, rate, points=60):
    """Overlapping Allan deviation of a rate signal. Returns tau [s], adev (same unit as omega)."""
    n = len(omega)
    theta = np.r_[np.zeros((1, omega.shape[1])), np.cumsum(omega, axis=0)] / rate
    ms = np.unique(np.logspace(0, np.log10(n // 10), points).astype(int))
    adev = np.empty((len(ms), omega.shape[1]))
    for j, m in enumerate(ms):
        tau = m / rate
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        adev[j] = np.sqrt((d ** 2).sum(axis=0) / (2 * tau ** 2 * (len(theta) - 2 * m)))
    return ms / rate, adev


def cmd_stationary(args):
    log = Log(args.log, args.rate, args.until)
    log.summary()
    keep = log.t >= args.skip
    t, gyro, temp = log.t[keep], log.gyro[keep], log.temp[keep]
    if log.gaps:
        print("NOTE: Allan deviation assumes continuous data, filled gaps make it slightly off")

    tau, adev = allan_deviation(gyro, log.rate)
    print("\naxis   ARW [°/√h]   bias instability [°/h]  (at τ [s])")
    for i, name in enumerate("XYZ"):
        arw = np.exp(np.interp(0.0, np.log(tau), np.log(adev[:, i]))) * 60  # σ(τ=1 s) in °/√s -> °/√h
        k = int(np.argmin(adev[:, i]))
        bi = adev[k, i] / 0.664 * 3600
        print(f"  {name}    {arw:8.3f}      {bi:10.2f}              ({tau[k]:.0f})")

    # bias vs time and temperature
    win = args.window * log.rate
    bias_t = moving_mean(gyro, win)
    temp_s = moving_mean(temp, win)
    half = int(win // 2)
    sl = slice(half, len(t) - half)  # drop edges where the moving mean is cut off
    bias0 = gyro[: int(args.bias_seconds * log.rate)].mean(axis=0)
    drift = np.cumsum(gyro - bias0, axis=0) / log.rate
    print(f"\nangle drift after {t[-1] - t[0]:.0f} s, bias from first {args.bias_seconds:g} s: "
          + "  ".join(f"{n} {drift[-1, i]:+.2f}°" for i, n in enumerate("XYZ")))
    print("bias vs temperature slope [°/s per °C]: ", end="")
    if np.ptp(temp_s[sl]) > 0.2:
        print("  ".join(f"{n} {np.polyfit(temp_s[sl], bias_t[sl, i], 1)[0]:+.4f}" for i, n in enumerate("XYZ")))
    else:
        print(f"n/a (temperature only changed {np.ptp(temp_s[sl]):.2f} °C)")

    if args.no_plot:
        return
    import matplotlib.pyplot as plt

    names, colors = "XYZ", ["#e74c3c", "#2ecc71", "#3498db"]
    fig1, ax = plt.subplots(figsize=(8, 6))
    for i in range(3):
        ax.loglog(tau, adev[:, i] * 3600, color=colors[i], marker=".", label=f"gyro {names[i]}")
    ax.set_xlabel("τ [s]")
    ax.set_ylabel("Allan deviation [°/h]")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    ax.set_title(f"Allan deviation: {args.log}")
    fig1.tight_layout()

    fig2, axs = plt.subplots(2, 2, figsize=(14, 8))
    for i in range(3):
        axs[0, 0].plot(t[sl], bias_t[sl, i] - bias_t[sl, i].mean(), color=colors[i], label=f"{names[i]}")
        axs[1, 0].plot(t, drift[:, i], color=colors[i], label=f"{names[i]}")
        axs[0, 1].plot(temp_s[sl], bias_t[sl, i] - bias_t[sl, i].mean(), color=colors[i], lw=0.8, label=f"{names[i]}")
    axs[1, 1].plot(t, temp, color="black", lw=0.5)
    axs[0, 0].set_title(f"gyro bias, {args.window:g} s moving mean, minus its average [°/s]")
    axs[0, 1].set_title("gyro bias vs temperature [°/s]")
    axs[0, 1].set_xlabel("temp [°C]")
    axs[1, 0].set_title(f"integrated angle drift, bias from first {args.bias_seconds:g} s [°]")
    axs[1, 1].set_title("die temperature [°C]")
    for ax in (axs[0, 0], axs[1, 0], axs[1, 1]):
        ax.set_xlabel("time [s]")
    for ax in axs.flat:
        ax.grid(True, ls=":", alpha=0.5)
    for ax in (axs[0, 0], axs[0, 1], axs[1, 0]):
        ax.legend(fontsize=8)
    fig2.suptitle(args.log)
    fig2.tight_layout()
    if args.save:
        base = args.save.rsplit(".", 1)[0]
        fig1.savefig(base + ".allan.png", dpi=150)
        fig2.savefig(base + ".drift.png", dpi=150)
    plt.show()


# ============================================================ calib6

FACE_MIN_ALIGN = 0.97  # hold counts as a face if gravity is within ~14° of a sensor axis
CALIB_COLORS = {"raw": "#a3a29c", "cal": "#2a78d6", "axes": ["#2a78d6", "#eb6834", "#1baf7a"]}


def face_name(g):
    """Which sensor axis points up, e.g. "+Z", and how well aligned (cosine, 1 = exactly)."""
    u = g / np.linalg.norm(g)
    i = int(np.argmax(np.abs(u)))
    return ("+" if u[i] > 0 else "-") + "XYZ"[i], abs(u[i])


def classify_faces(holds, acc):
    """Set h["face"] to "+X".. for holds square to an axis, None for check holds."""
    for h, a in zip(holds, acc):
        name, align = face_name(a)
        h["face"] = name if align >= FACE_MIN_ALIGN else None
    faces = {}
    for h in holds:
        if h["face"]:
            faces.setdefault(h["face"], []).append(h["acc"])
    return {k: np.mean(v, axis=0) for k, v in faces.items()}


def fit_box(faces):
    """Classic 6-position calibration: a_meas = M @ a_true + b, with a_true = ±1 g along each axis.

    faces: {"+X": mean accel, ...}, all six needed. Column i of M is half the difference between
    axis i up and down; each up/down pair also gives an estimate of b, and b is their mean.
    Off-diagonal terms of M mix axis misalignment with how square the faces sat on the table.
    """
    M = np.column_stack([(faces["+" + n] - faces["-" + n]) / 2 for n in "XYZ"])
    b = np.mean([(faces["+" + n] + faces["-" + n]) / 2 for n in "XYZ"], axis=0)
    return M, b


def fit_sphere(acc, iters=100):
    """Per-axis offset b and scale s so that |(a - b) / s| = 1 g for every hold (Gauss-Newton).

    Uses only the length of gravity, so the holds don't need to be square to the table. Needs at
    least 6 holds spread over all directions.
    """
    b, s = np.zeros(3), np.ones(3)
    for _ in range(iters):
        u = (acc - b) / s
        n = np.linalg.norm(u, axis=1)[:, None]
        J = np.hstack([-u / (n * s), -u ** 2 / (n * s)])  # d|u| / d(b, s)
        step = np.linalg.lstsq(J, -(n[:, 0] - 1), rcond=None)[0]
        b, s = b + step[:3], s + step[3:]
        if np.linalg.norm(step) < 1e-12:
            break
    return b, s


def cmd_calib6(args):
    """Accelerometer 6-position calibration and gyro g-sensitivity.

    Record with `record`, hold the board still for ~10 s with each sensor axis pointing up and
    then down (+X -X +Y -Y +Z -Z, any order), and rotate between holds. A few extra holds at
    odd angles (e.g. 45° on a book) are not used for the fit and serve as an independent check.
    """
    import json

    log = Log(args.log, args.rate, args.until)
    log.summary()
    gyro_bias = np.median(log.gyro, axis=0)  # a calibration run is mostly holds
    segs = find_still(log.gyro - gyro_bias, log.rate, args.still_thresh, args.min_still, args.trim)
    if not segs:
        sys.exit("no still holds found")

    holds = []
    for s, e in segs:
        a = log.acc[s:e].mean(axis=0)
        holds.append(dict(t0=log.t[s], t1=log.t[e - 1], acc=a, acc_std=log.acc[s:e].std(axis=0),
                          gyro=log.gyro[s:e].mean(axis=0), temp=log.temp[s:e].mean()))
    acc = np.array([h["acc"] for h in holds])

    # A big offset makes raw faces look tilted (0.4 g on Z tilts "+X up" by ~20°), so first take
    # the best aligned hold per face, fit, and then classify again on calibrated vectors.
    best = {}
    for h in holds:
        name, align = face_name(h["acc"])
        if align > best.get(name, (0, None))[0]:
            best[name] = (align, h["acc"])
    missing = [f"{sg}{n}" for n in "XYZ" for sg in "+-" if f"{sg}{n}" not in best]
    if missing:
        sys.exit(f"{len(holds)} holds found, faces missing: {' '.join(missing)} "
                 f"(hold each axis up and down for ≥ {args.min_still:g} s, or lower --still-thresh)")
    M, b = fit_box({k: a for k, (_, a) in best.items()})
    faces = classify_faces(holds, (acc - b) @ np.linalg.inv(M).T)
    if len(faces) < 6:
        sys.exit(f"after a first fit only faces {' '.join(sorted(faces))} are square to an axis")

    M, b = fit_box(faces)
    Minv = np.linalg.inv(M)
    face_acc = np.array([h["acc"] for h in holds if h["face"]])
    sb, ss = fit_sphere(face_acc)

    raw_err = (np.linalg.norm(acc, axis=1) - 1) * 1000  # mg
    box_err = (np.linalg.norm((acc - b) @ Minv.T, axis=1) - 1) * 1000
    sph_err = (np.linalg.norm((acc - sb) / ss, axis=1) - 1) * 1000
    is_check = np.array([h["face"] is None for h in holds])
    temps = np.array([h["temp"] for h in holds])

    print(f"\n{len(holds)} holds ({len(holds) - is_check.sum()} on faces, {is_check.sum()} check)"
          f", temp {temps.min():.2f}..{temps.max():.2f} °C")
    print(" hold   t[s]   dur  face  |      acc mean [g]       noise [mg] |  |a|-1 [mg]: raw    box  sphere")
    for k, h in enumerate(holds):
        print(f"{k:4d} {h['t0']:7.1f} {h['t1'] - h['t0']:5.1f}  {h['face'] or 'chk':>4}  |"
              f" {h['acc'][0]:+6.3f} {h['acc'][1]:+6.3f} {h['acc'][2]:+6.3f}   {np.linalg.norm(h['acc_std']) * 1000:5.1f}    |"
              f"           {raw_err[k]:+7.1f} {box_err[k]:+6.1f} {sph_err[k]:+6.1f}")

    def rms(x):
        return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")

    print("\naxis | box: offset [mg]  scale err [%] | sphere: offset [mg]  scale err [%]")
    for i, n in enumerate("XYZ"):
        print(f"  {n}  |      {b[i] * 1000:+8.1f}     {(M[i, i] - 1) * 100:+7.2f}    |"
              f"        {sb[i] * 1000:+8.1f}     {(ss[i] - 1) * 100:+7.2f}")
    # angle between each measured axis and the ideal one: misalignment plus how square the faces sat
    cols = M / np.linalg.norm(M, axis=0)
    off = [np.degrees(np.arccos(np.clip(cols[i, i], -1, 1))) for i in range(3)]
    print("box: axis tilt vs ideal [°]: " + "  ".join(f"{n} {off[i]:.2f}" for i, n in enumerate("XYZ"))
          + "  (chip misalignment + faces not square to the table)")
    print(f"|a| error RMS [mg]: raw {rms(raw_err):.1f}, box {rms(box_err[~is_check]):.1f} on faces"
          f" / {rms(box_err[is_check]):.1f} on check holds, sphere {rms(sph_err[~is_check]):.1f}"
          f" / {rms(sph_err[is_check]):.1f}")

    # Gyro bias vs gravity: gyro = c + G @ g_true. Temperature drift during the run leaks in.
    g_true = (acc - b) @ Minv.T
    X = np.hstack([np.ones((len(holds), 1)), g_true])
    coef = np.linalg.lstsq(X, np.array([h["gyro"] for h in holds]), rcond=None)[0]
    G = coef[1:].T  # G[i, j]: gyro axis i per g along axis j
    print("\ngyro bias vs gravity [°/s per g] (rows: gyro axis, columns: gravity axis)")
    for i, n in enumerate("XYZ"):
        print(f"  {n}  " + "  ".join(f"{G[i, j]:+.4f}" for j in range(3)))
    if np.ptp(temps) > 0.5:
        print(f"NOTE: temperature moved {np.ptp(temps):.1f} °C between holds, part of this may be temperature")

    out = args.log.rsplit(".", 1)[0] + ".calib.json"
    with open(out, "w") as f:
        json.dump({
            "log": args.log,
            "model": "a_true = inverse @ (a_meas - offset), a in g",
            "offset": b.tolist(), "matrix": M.tolist(), "inverse": Minv.tolist(),
            "sphere_offset": sb.tolist(), "sphere_scale": ss.tolist(),
            "gyro_g_sensitivity_dps_per_g": G.tolist(),
            "temp_c": float(temps.mean()),
        }, f, indent=2)
    print(f"calibration saved to {out}")

    if args.no_plot:
        return
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    labels = [h["face"] or "chk" for h in holds]

    fig1, ax = plt.subplots(figsize=(13, 5))
    for i, n in enumerate("XYZ"):
        ax.plot(log.t, log.acc[:, i], color=CALIB_COLORS["axes"][i], lw=1.2, label=f"accel {n}")
    for h, lab in zip(holds, labels):
        ax.axvspan(h["t0"], h["t1"], color="#e4e3df", lw=0, zorder=0)
        ax.annotate(lab, ((h["t0"] + h["t1"]) / 2, 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=12, color="#52514e")
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("acceleration [g]")
    ax.grid(True, axis="y", ls=":", alpha=0.6)
    ax.legend(loc="lower right", ncols=3, frameon=False)
    ax.set_title("6-position calibration run (shaded = detected holds)", pad=24)
    fig1.tight_layout()

    fig2, axs = plt.subplots(1, 3, figsize=(16, 5.5), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    x = np.arange(len(holds))
    axs[0].bar(x - 0.2, raw_err, 0.4, color=CALIB_COLORS["raw"], label="raw")
    axs[0].bar(x + 0.2, box_err, 0.4, color=CALIB_COLORS["cal"], label="calibrated (box)")
    axs[0].axhline(0, color="#52514e", lw=0.8)
    axs[0].set_xticks(x, labels)
    axs[0].set_ylabel("|a| − 1 g  [mg]")
    axs[0].set_title(f"gravity length error per hold\nRMS raw {rms(raw_err):.0f} mg → calibrated "
                     f"{rms(box_err):.1f} mg")
    axs[0].legend(frameon=False)
    axs[1].bar(list("XYZ"), b * 1000, 0.6, color=CALIB_COLORS["axes"])
    axs[1].set_title("offset [mg]")
    axs[2].bar(list("XYZ"), (np.diag(M) - 1) * 100, 0.6, color=CALIB_COLORS["axes"])
    axs[2].set_title("scale error [%]")
    for ax in axs:
        ax.grid(True, axis="y", ls=":", alpha=0.6)
        ax.axhline(0, color="#52514e", lw=0.8)
    fig2.tight_layout()

    if args.save:
        base = args.save.rsplit(".", 1)[0]
        fig1.savefig(base + ".calib-run.png", dpi=200)
        fig2.savefig(base + ".calib-result.png", dpi=200)
    plt.show()


# ============================================================ main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="log every IMU sample to CSV")
    r.add_argument("out")
    r.add_argument("--port", default=navlink.PORT)
    r.add_argument("-d", "--duration", type=float, default=0, help="stop after this many seconds (0 = Ctrl+C)")
    r.set_defaults(func=cmd_record)

    def common(sp):
        sp.add_argument("log")
        sp.add_argument("--rate", default="host",
                        help="sample rate: host (fit to PC clock), mcu, nominal, or a number in Hz")
        sp.add_argument("--no-plot", action="store_true")
        sp.add_argument("--save", metavar="PNG", help="also save the plot(s)")
        sp.add_argument("--until", type=float, help="only use the log up to this many seconds (host time)")

    a = sub.add_parser("analyze", help="rotate-and-hold runs: find still segments, compare angles")
    common(a)
    a.add_argument("--bias-seconds", type=float, default=5,
                   help="gyro bias from the first N s of the log (0 = use boot bias)")
    a.add_argument("--still-thresh", type=float, default=1.0, help="°/s, below this counts as still")
    a.add_argument("--min-still", type=float, default=1.0, help="s, shortest still segment kept")
    a.add_argument("--trim", type=float, default=0.3, help="s cut from both ends of each still segment")
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser("stationary", help="long still logs: Allan deviation, bias drift vs time and temp")
    common(s)
    s.add_argument("--skip", type=float, default=0, help="ignore the first N s (warm-up)")
    s.add_argument("--window", type=float, default=10, help="s, moving mean window for bias vs time")
    s.add_argument("--bias-seconds", type=float, default=10, help="bias window for the angle drift plot")
    s.set_defaults(func=cmd_stationary)

    c = sub.add_parser("calib6", help="6-position accel calibration (+ gyro g-sensitivity)")
    common(c)
    c.add_argument("--still-thresh", type=float, default=1.0, help="°/s, below this counts as still")
    c.add_argument("--min-still", type=float, default=3.0, help="s, shortest hold kept")
    c.add_argument("--trim", type=float, default=0.5, help="s cut from both ends of each hold")
    c.set_defaults(func=cmd_calib6)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
