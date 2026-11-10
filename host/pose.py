"""Orientation and position of the ToF sensor for every frame of a tripod recording (record.py).

    python pose.py scan.navrec                      # writes scan.pose.csv and shows check plots
    python pose.py scan.navrec --offset 0.085 0 0.03 --no-plot

Model: the sensor sits on a pan/tilt tripod head. It only rotates, so the ToF moves on a sphere
around the pivot (where the pan and tilt axes meet) at a fixed body-frame offset (--offset).

How the orientation is found:
  1. Still moments (gyro and accel both steady, judged on raw values) give the gyro bias at
     that time. The bias is blended linearly between still moments, held before the first and
     after the last. At least one still moment is needed: the boot bias from the MCU is not
     trusted (it is often invalid).
  2. Gyro Z is divided by GYRO_Z_SCALE (measured, see below).
  3. Accel gets the 6-position calibration (imu_experiment.py calib6), then the still moments
     refine its offset so that |g| = 1 there. One still only fixes the offset along gravity
     (mostly Z), which is the big one.
  4. The gyro is integrated into quaternions. Accel slowly pulls pitch and roll toward gravity
     (time constant --tau, ignored while |a| is far from 1 g). Gravity says nothing about
     heading, so yaw is from the gyro only. Run forward and backward in time and averaged, so
     the correction adds no lag.
  5. Each ToF frame takes the orientation at the middle of its measurement, --tof-shift-ms
     before its data-ready edge.

Output, one row per ToF frame (comment lines start with '#'):
    frame, vl53_seq, t_s, still, gap, qw, qx, qy, qz, tof_x_m, tof_y_m, tof_z_m, yaw_deg, pitch_deg, roll_deg
  q (w, x, y, z) rotates body vectors into the world: v_world = q * v_body * q^-1.
  Body frame = IMU axes: +X is where the ToF looks, +Z is "up" of the ToF grid as main.py
  shows it (row 0 of the 8x8 grid is toward +Z).
  World frame: +Z up (gravity), +X = horizontal direction the ToF faced at the first frame,
  origin = pivot. tof_* = q * offset, the ToF position in metres.
  yaw: left turn positive; pitch: looking up positive; roll: about body +X. For people
  reading only, use q for any math.
  still = frame is inside a still moment, gap = IMU data was missing around the frame.
"""

import argparse
import datetime
import json
import pathlib
import sys

import numpy as np

import navlink
from imu_experiment import moving_mean, quat_to_euler

# The gyro reads 0.2441% too much rotation on Z: 360° pan tests (360.1.csv) integrated with the
# host-clock sample rate gave 360.81° and 360.95°. That clock is the true one, so this is the
# gyro's own scale error, not a timing effect. X and Y were never measured.
GYRO_Z_SCALE = 1.002441
DEFAULT_CALIB = pathlib.Path(__file__).resolve().parent / "from_pc" / "6pos.calib.json"
DEFAULT_OFFSET_M = (0.085, 0.0, 0.0)  # ToF 8.5 cm forward of the pan and tilt axes, height unknown

STILL_WINDOW_S = 0.25
STILL_GYRO_DPS = 1.0   # smoothed |gyro - rough bias| below this
STILL_ACC_G = 0.03     # accel spread (std, all axes) below this
STILL_MIN_S = 1.0
STILL_TRIM_S = 0.2
ACC_TRUST_G = 0.02     # accel fully trusted within this of 1 g, ignored past 2.5x this
MAX_TRUSTED_GAP_S = 0.2

COLUMNS = ("frame,vl53_seq,t_s,still,gap,qw,qx,qy,qz,tof_x_m,tof_y_m,tof_z_m,"
           "yaw_deg,pitch_deg,roll_deg")


# ============================================================ quaternions (w, x, y, z)

def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qrot(q, v):
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4)."""
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    return qmul(qmul(q, qv), qconj(q))[..., 1:]


def qslerp(a, b, t):
    """Spherical interpolation between unit quaternions a and b, t in [0, 1] (row-wise)."""
    d = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(d < 0, -b, b)  # take the short way round
    d = np.abs(d)
    t = np.asarray(t, dtype=float).reshape(d.shape)
    theta = np.arccos(np.clip(d, -1, 1))
    s = np.sin(theta)
    small = s < 1e-9
    wa = np.where(small, 1 - t, np.sin((1 - t) * theta) / np.where(small, 1, s))
    wb = np.where(small, t, np.sin(t * theta) / np.where(small, 1, s))
    q = wa * a + wb * b
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_level(up_body):
    """Body -> world quaternion with world Z = measured up and body X pointing to world yaw 0."""
    z = up_body / np.linalg.norm(up_body)
    x = np.array([1.0, 0.0, 0.0]) - z[0] * z  # body X with its vertical part removed
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])  # rows: world axes in body coordinates, so v_world = R @ v_body
    w = np.sqrt(max(1e-12, 1 + np.trace(R))) / 2
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                     (R[1, 0] - R[0, 1]) / (4 * w)])


# ============================================================ loading

class Recording:
    def __init__(self, path):
        meta, pkts, self.marks = navlink.load_recording(path)
        self.start_wall = meta.get("start_wall", "")
        self.frames = pkts
        imu_pkts = [p for p in pkts if p.n_imu > 0]
        if not imu_pkts:
            sys.exit(f"{path}: no IMU data in this recording (IMU disconnected?)")
        p0 = imu_pkts[0]
        self.acc_scale, self.gyro_scale = p0.accel_scale, p0.gyro_scale

        seq = np.concatenate([p.imu_seq0 + np.arange(p.n_imu) for p in imu_pkts]).astype(np.int64)
        raw = np.concatenate([p.imu for p in imu_pkts]).astype(np.float64)
        end_seq = np.array([p.imu_seq0 + p.n_imu - 1 for p in imu_pkts], dtype=np.int64)
        end_mcu = np.array([p.imu_t_us for p in imu_pkts]) * 1e-6
        end_host = np.array([p.host_t for p in imu_pkts])
        nominal = float(p0.imu_rate_hz)

        # The MCU clock maps sample index <-> ToF frame times (both stamped by the MCU). Fit it
        # over the whole recording so per-packet jitter averages out.
        if len(end_seq) >= 10 and np.ptp(end_mcu) > 1:
            self.rate_mcu = np.polyfit(end_mcu, end_seq, 1)[0]
        else:
            self.rate_mcu = nominal
        # Gaps: a seq jump says how many samples are missing. A FIFO reset (gap flag, seq keeps
        # counting) does not, so measure that one on the MCU clock.
        pos = seq - seq[0]
        gap_flags = np.zeros(len(seq), dtype=bool)
        row = 0
        for k, p in enumerate(imu_pkts):
            if k > 0 and p.flags & navlink.FLAG_IMU_GAP_BEFORE:
                gap_flags[row] = True
            row += p.n_imu
        missing = np.zeros(len(seq), dtype=np.int64)
        for i in np.flatnonzero(gap_flags | np.r_[False, np.diff(seq) != 1]):
            n = int(seq[i] - seq[i - 1] - 1)
            if gap_flags[i]:
                k = np.searchsorted(np.cumsum([p.n_imu for p in imu_pkts]), i, side="right")
                t_i = end_mcu[k] - (end_seq[k] - seq[i]) / self.rate_mcu
                t_prev = end_mcu[k - 1]
                n = max(n, int(round((t_i - t_prev) * self.rate_mcu)) - 1)
            missing[i] = max(n, 0)
        pos = np.arange(len(seq)) + np.cumsum(missing)
        self.gaps = [(int(pos[i] - missing[i]), int(missing[i])) for i in np.flatnonzero(missing)]

        # Sample index on the MCU clock, fitted on the gap-free timeline
        end_pos = pos[np.searchsorted(seq, end_seq)]
        if len(end_pos) >= 10 and np.ptp(end_mcu) > 1:
            self.mcu_fit = np.polyfit(end_mcu, end_pos, 1)
            self.rate_mcu = self.mcu_fit[0]
        else:
            self.mcu_fit = np.array([nominal, end_pos[0] - nominal * end_mcu[0]])
        # True sample rate for integration: the host clock. USB jitter averages out over a
        # long recording. Short ones fall back to nominal.
        if len(end_pos) >= 10 and np.ptp(end_host) > 5:
            self.rate = np.polyfit(end_host, end_pos, 1)[0]
        else:
            self.rate = nominal

        n = int(pos[-1]) + 1
        full = np.arange(n)
        interp = lambda a: np.column_stack([np.interp(full, pos, a[:, k]) for k in range(a.shape[1])])
        self.acc_raw = interp(raw[:, 0:3]) / self.acc_scale    # g, uncalibrated
        self.gyro_raw = interp(raw[:, 3:6]) / self.gyro_scale  # °/s, no bias removed
        self.temp = np.interp(full, pos, navlink.temp_to_c(raw[:, 6]))
        self.filled = np.ones(n, dtype=bool)
        self.filled[pos] = False
        self.t = full / self.rate

    def mcu_to_index(self, t_mcu_s):
        return np.polyval(self.mcu_fit, t_mcu_s)


# ============================================================ calibration

def find_still(rec):
    """Index ranges [start, end) where gyro and accel are both steady."""
    w = max(1, int(STILL_WINDOW_S * rec.rate))
    acc_std = np.sqrt(np.sum(moving_mean(rec.acc_raw ** 2, w) - moving_mean(rec.acc_raw, w) ** 2, axis=1).clip(0))
    quiet = acc_std < STILL_ACC_G
    # rough bias from the quietest samples, then the real test is on the smoothed rate
    rough = np.median(rec.gyro_raw[quiet], axis=0) if quiet.sum() > w else np.median(rec.gyro_raw, axis=0)
    rate = np.linalg.norm(moving_mean(rec.gyro_raw - rough, w), axis=1)
    still = quiet & (rate < STILL_GYRO_DPS) & ~rec.filled
    edges = np.flatnonzero(np.diff(np.r_[0, still.astype(int), 0]))
    trim = int(STILL_TRIM_S * rec.rate)
    segs = []
    for s, e in zip(edges[::2], edges[1::2]):
        s, e = s + trim, e - trim
        if (e - s) / rec.rate >= STILL_MIN_S:
            segs.append((s, e))
    return segs


def gyro_bias_track(rec, segs):
    """Per-sample gyro bias, blended linearly between still moments."""
    centers = np.array([(s + e - 1) / 2 for s, e in segs])
    biases = np.array([rec.gyro_raw[s:e].mean(axis=0) for s, e in segs])
    idx = np.arange(len(rec.t))
    track = np.column_stack([np.interp(idx, centers, biases[:, k]) for k in range(3)])
    return track, biases


def refine_accel(acc_still, ridge=0.05):
    """Small extra offset so that |a| = 1 g at the still moments.

    Each still constrains only the offset along its gravity direction; the ridge keeps the
    directions no still moment sees at zero.
    """
    b = np.zeros(3)
    for _ in range(5):
        u = acc_still - b
        n = np.linalg.norm(u, axis=1)
        J = -u / n[:, None]
        A = np.vstack([J, np.sqrt(ridge) * np.eye(3)])
        r = np.concatenate([-(n - 1), -np.sqrt(ridge) * b])
        b = b + np.linalg.lstsq(A, r, rcond=None)[0]
    return b


# ============================================================ orientation

def filter_pass(omega, acc, dt, q0, kp):
    """Gyro integration with a slow pull of the tilt toward gravity. Returns body -> world q per sample.

    omega in rad/s (body), acc in g (body, calibrated), kp in 1/s.
    """
    n = len(omega)
    out = np.empty((n, 4))
    q = np.array(q0, dtype=float)
    anorm = np.linalg.norm(acc, axis=1)
    trust = np.clip(1 - (np.abs(anorm - 1) - ACC_TRUST_G) / (1.5 * ACC_TRUST_G), 0, 1)
    for i in range(n):
        out[i] = q
        w = omega[i]
        if trust[i] > 0:
            up_meas = acc[i] / anorm[i]
            qw, qx, qy, qz = q
            # world Z seen in the body frame (third row of the rotation matrix)
            up_est = np.array([2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)])
            w = w + kp * trust[i] * np.cross(up_meas, up_est)
        mag = np.sqrt(w @ w)
        if mag > 1e-12:
            half = 0.5 * mag * dt
            d = np.r_[np.cos(half), w * (np.sin(half) / mag)]
            q = qmul(q, d)
            q /= np.linalg.norm(q)
    return out


def orientation(omega, acc, dt, q0, tau):
    kp = 1.0 / tau
    fwd = filter_pass(omega, acc, dt, q0, kp)
    # Backward: time reversed, starting where the forward pass ended. Going from sample i to
    # i - 1 undoes the rate of sample i - 1.
    back_omega = -np.roll(omega[::-1], -1, axis=0)
    bwd = filter_pass(back_omega, acc[::-1], dt, fwd[-1], kp)[::-1]
    return qslerp(fwd, bwd, np.full(len(omega), 0.5))


def principal_axis(v):
    """Unit direction that best fits the vectors v (sign: largest component positive)."""
    _, _, vt = np.linalg.svd(v, full_matrices=False)
    a = vt[0]
    return a if a[np.argmax(np.abs(a))] > 0 else -a


# ============================================================ main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording")
    p.add_argument("-o", "--out", help="output CSV (default: <recording>.pose.csv)")
    p.add_argument("--calib", default=str(DEFAULT_CALIB), help="accel calibration JSON from imu_experiment.py calib6")
    p.add_argument("--offset", type=float, nargs=3, default=DEFAULT_OFFSET_M, metavar=("X", "Y", "Z"),
                   help="ToF position from the pivot in the body frame, metres")
    p.add_argument("--gyro-z-scale", type=float, default=GYRO_Z_SCALE, help="gyro Z reads this many times too much")
    p.add_argument("--tau", type=float, default=1.0, help="s, how slowly accel corrects pitch and roll")
    p.add_argument("--tof-shift-ms", type=float, default=33.0,
                   help="ms before the data-ready edge where a ToF frame's measurement is centred")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--save", metavar="PNG", help="also save the check plot")
    args = p.parse_args()

    rec = Recording(args.recording)
    print(f"{len(rec.t)} IMU samples, {rec.t[-1]:.1f} s, rate {rec.rate:.3f} Hz (host clock), "
          f"{rec.rate_mcu:.3f} Hz on the MCU clock, temp {rec.temp.min():.2f}..{rec.temp.max():.2f} °C")
    if rec.gaps:
        print(f"WARNING: {len(rec.gaps)} IMU gaps, {int(rec.filled.sum())} samples filled by interpolation")

    # --- gyro bias
    segs = find_still(rec)
    if not segs:
        sys.exit("no still moments found, so the gyro bias is unknown. Hold the tripod still for a "
                 "couple of seconds at the start (and ideally between sweeps).")
    bias_track, biases = gyro_bias_track(rec, segs)
    print(f"\n{len(segs)} still moments, gyro bias at each [°/s]:")
    for (s, e), b in zip(segs, biases):
        print(f"  {rec.t[s]:6.1f} s  {(e - s) / rec.rate:4.1f} s   {b[0]:+7.3f} {b[1]:+7.3f} {b[2]:+7.3f}")
    if len(segs) == 1:
        print("  only one still moment: bias drift is not tracked")

    # --- accel
    try:
        with open(args.calib) as f:
            c = json.load(f)
    except FileNotFoundError:
        sys.exit(f"{args.calib} not found: run imu_experiment.py calib6 on a 6-position recording, or pass --calib")
    acc = (rec.acc_raw - np.array(c["offset"])) @ np.array(c["inverse"]).T
    acc_still = np.array([acc[s:e].mean(axis=0) for s, e in segs])
    extra = refine_accel(acc_still)
    before = (np.linalg.norm(acc_still, axis=1) - 1) * 1000
    acc = acc - extra
    after = (np.linalg.norm(acc_still - extra, axis=1) - 1) * 1000
    print(f"\naccel: {args.calib}, extra offset from still moments [mg] "
          + " ".join(f"{v * 1000:+.1f}" for v in extra))
    print("  |g| - 1 at still moments [mg]: before " + " ".join(f"{v:+.1f}" for v in before)
          + "  after " + " ".join(f"{v:+.1f}" for v in after))

    # --- orientation
    gyro = rec.gyro_raw - bias_track
    gyro[:, 2] /= args.gyro_z_scale
    omega = np.radians(gyro)
    s0, e0 = segs[0]
    # start level from the first still moment if the recording starts with one, else first 0.25 s
    up0 = acc[s0:e0].mean(axis=0) if s0 < rec.rate else acc[: int(STILL_WINDOW_S * rec.rate)].mean(axis=0)
    q = orientation(omega, acc, 1.0 / rec.rate, quat_level(up0), args.tau)

    # --- per ToF frame
    frames = rec.frames
    shift = args.tof_shift_ms * 1e-3
    fidx = rec.mcu_to_index(np.array([fr.vl53_t_us for fr in frames]) * 1e-6 - shift)
    ok = (fidx >= 0) & (fidx <= len(rec.t) - 1)
    if not ok.all():
        print(f"\n{(~ok).sum()} ToF frames outside the IMU data, skipped")
    i0 = np.floor(fidx[ok]).astype(int)
    i1 = np.minimum(i0 + 1, len(rec.t) - 1)
    qf = qslerp(q[i0], q[i1], fidx[ok] - i0)

    # yaw 0 = where the ToF faced at the first frame
    fwd0 = qrot(qf[0], np.array([1.0, 0.0, 0.0]))
    yaw_start = np.arctan2(fwd0[1], fwd0[0])
    qyaw = np.array([np.cos(-yaw_start / 2), 0.0, 0.0, np.sin(-yaw_start / 2)])
    qf = qmul(np.broadcast_to(qyaw, qf.shape), qf)
    q_all = qmul(np.broadcast_to(qyaw, q.shape), q)

    pos = qrot(qf, np.broadcast_to(np.array(args.offset), (len(qf), 3)))
    yaw, pitch, roll = quat_to_euler(qf)
    pitch = -pitch  # ZYX pitch is nose down; flip so looking up is positive

    still_mask = np.zeros(len(rec.t), dtype=bool)
    for s, e in segs:
        still_mask[s:e] = True
    w_gap = int(MAX_TRUSTED_GAP_S * rec.rate)
    gap_near = moving_mean(rec.filled.astype(float), max(1, w_gap)) > 0
    seqs = np.array([fr.vl53_seq for fr in frames])[ok]
    tf = fidx[ok] / rec.rate

    out = args.out or args.recording.rsplit(".", 1)[0] + ".pose.csv"
    with open(out, "w") as f:
        f.write("# nav-pose v1\n")
        f.write(f"# recording={args.recording}\n")
        f.write(f"# made={datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write("# q rotates body to world. body: +X ToF look direction, +Z grid up. "
                "world: +Z up, +X initial look direction, origin pivot\n")
        f.write(f"# offset_m={','.join(str(v) for v in args.offset)}\n")
        f.write(f"# tof_shift_ms={args.tof_shift_ms}\n")
        f.write(f"# columns={COLUMNS}\n")
        for k in range(len(qf)):
            f.write(f"{np.flatnonzero(ok)[k]},{seqs[k]},{tf[k]:.4f},{int(still_mask[i0[k]])},{int(gap_near[i0[k]])},"
                    + ",".join(f"{v:.7f}" for v in qf[k]) + ","
                    + ",".join(f"{v:.5f}" for v in pos[k]) + ","
                    + f"{yaw[k]:.3f},{pitch[k]:.3f},{roll[k]:.3f}\n")
    print(f"\n{len(qf)} frames saved to {out}")

    # --- checks
    print(f"\nyaw {yaw.min():+.1f}..{yaw.max():+.1f}°, pitch {pitch.min():+.1f}..{pitch.max():+.1f}°, "
          f"roll {roll.min():+.2f}..{roll.max():+.2f}° (spread {np.ptp(roll):.2f}°)")
    for s, e in segs:
        k = (tf >= rec.t[s]) & (tf <= rec.t[e - 1])
        if k.any():
            print(f"  still at {rec.t[s]:6.1f} s: yaw {yaw[k].mean():+8.2f}°  pitch {pitch[k].mean():+6.2f}°"
                  f"  roll {roll[k].mean():+5.2f}°")

    # Tilt axis (body frame) from samples turning mostly about a horizontal axis, pan axis
    # (world frame) from samples turning mostly about body Y's perpendicular. A level tripod
    # has the pan axis on world Z; if gravity (accel) were off, this shows the angle.
    w_body = omega
    w_world = qrot(q_all, w_body)
    speed = np.linalg.norm(w_body, axis=1)
    moving = speed > np.radians(5)
    tilt_like = moving & (np.abs(w_world[:, 2]) < 0.2 * speed)
    pan_like = moving & (np.abs(w_body[:, 1]) < 0.2 * speed)
    if tilt_like.sum() > 50:
        a = principal_axis(w_body[tilt_like])
        ang = np.degrees(np.arccos(np.clip(abs(a[1]), 0, 1)))
        print(f"\ntilt axis in body frame: ({a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f}), {ang:.2f}° from body Y "
              f"({tilt_like.sum() / rec.rate:.1f} s of tilting)")
    if pan_like.sum() > 50:
        a = principal_axis(w_world[pan_like])
        ang = np.degrees(np.arccos(np.clip(abs(a[2]), 0, 1)))
        print(f"pan axis in world frame: ({a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f}), {ang:.2f}° from vertical "
              f"({pan_like.sum() / rec.rate:.1f} s of panning)")

    if args.no_plot and not args.save:
        return
    import matplotlib.pyplot as plt

    # accel-only tilt per sample, for comparison with the filter
    an = acc / np.linalg.norm(acc, axis=1)[:, None]
    acc_pitch = np.degrees(np.arcsin(np.clip(an[:, 0], -1, 1)))  # +X up = looking up
    acc_roll = np.degrees(np.arctan2(an[:, 1], an[:, 2]))
    _, s_pitch, s_roll = quat_to_euler(q_all)
    s_pitch = -s_pitch  # looking up positive, as in the CSV

    fig, axs = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    for ax in axs:
        for s, e in segs:
            ax.axvspan(rec.t[s], rec.t[e - 1], color="#1baf7a", alpha=0.15, lw=0)
        ax.grid(True, ls=":", alpha=0.5)
    axs[0].plot(tf, yaw, color="#2a78d6")
    axs[0].set_ylabel("yaw [°]")
    axs[0].set_title(f"{args.recording}: pose per ToF frame (green = still moments)")
    axs[1].plot(rec.t, acc_pitch, color="#a3a29c", lw=0.6, label="accel only")
    axs[1].plot(rec.t, s_pitch, color="#eb6834", label="filter")
    axs[1].set_ylabel("pitch [°]")
    axs[1].legend(loc="upper right")
    axs[2].plot(rec.t, acc_roll, color="#a3a29c", lw=0.6, label="accel only")
    axs[2].plot(rec.t, s_roll, color="#eb6834", label="filter")
    axs[2].set_ylim(np.median(s_roll) - 5, np.median(s_roll) + 5)
    axs[2].set_ylabel("roll [°]")
    axs[2].legend(loc="upper right")
    for k, (name, col) in enumerate(zip("XYZ", ["#2a78d6", "#eb6834", "#1baf7a"])):
        axs[3].plot(rec.t, bias_track[:, k] - biases[0, k], color=col, label=f"{name} ({biases[0, k]:+.2f} at first still)")
    axs[3].set_ylabel("gyro bias change [°/s]")
    axs[3].set_xlabel("t [s]")
    axs[3].legend(loc="upper right")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()
