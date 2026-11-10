"""Hand-held experiment: orientation like pose.py, plus position from double-integrated accel.

    python track.py walk.navrec                     # writes walk.track.csv and shows check plots
    python track.py walk.navrec --no-plot

Expect drift: 1 mg of leftover accel error is ~0.5 m after 10 s. What keeps it in check is that
the sensor is known to stand still during still moments (same detection as pose.py): velocity is
forced to zero there, and the velocity error built up over each moving stretch is removed as a
straight ramp so it ends at zero too (same as a constant accel bias over that stretch). So
results are only as good as the still moments: hold the sensor still for a second or two often.
A moving stretch at the end of the recording with no still moment after it is integrated
without that correction and drifts freely.

Orientation, frames and calibration: see pose.py. Output has the same columns as pose.py, but
tof_x/y/z is the tracked position (origin = where the IMU was at the start, world axes as
pose.py), plus the ToF offset from the IMU if --offset is given.
"""

import argparse

import numpy as np

import pose

G = 9.80665


def integrate_with_stops(acc_world, still, dt):
    """Velocity and position from world-frame linear accel [m/s²], zero velocity while still.

    Returns (vel, pos, corrections): corrections lists (start, end, implied accel bias [m/s²],
    corrected) per moving stretch.
    """
    n = len(acc_world)
    vel = np.zeros((n, 3))
    edges = np.flatnonzero(np.diff(np.r_[0, (~still).astype(int), 0]))
    corrections = []
    for s, e in zip(edges[::2], edges[1::2]):  # moving stretches [s, e)
        v = np.cumsum(acc_world[s:e], axis=0) * dt
        ends_still = e < n
        if ends_still:
            # still again at e: whatever velocity is left is error, take it out as a ramp
            ramp = np.arange(1, e - s + 1)[:, None] / (e - s)
            bias = v[-1] / ((e - s) * dt)
            v = v - ramp * v[-1]
        else:
            bias = np.full(3, np.nan)
        vel[s:e] = v
        corrections.append((s, e, bias, ends_still))
    pos = np.cumsum(vel, axis=0) * dt
    return vel, pos, corrections


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pose.add_common_args(p, ".track.csv")
    p.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                   help="ToF position from the IMU in the body frame, metres")
    args = p.parse_args()

    sol = pose.solve(args)
    rec, dt = sol.rec, 1.0 / sol.rec.rate

    # accel measures "up" (reaction to gravity) plus motion; remove 1 g up in the world frame
    acc_world = (pose.qrot(sol.q, sol.acc) - np.array([0.0, 0.0, 1.0])) * G
    acc_world[sol.still] = 0.0
    vel, pos, corr = integrate_with_stops(acc_world, sol.still, dt)

    print("\nmoving stretches (velocity error removed at the next still moment):")
    for s, e, bias, ok in corr:
        span = f"  {rec.t[s]:6.1f}..{rec.t[min(e, len(rec.t) - 1)]:6.1f} s ({(e - s) * dt:5.1f} s)"
        if ok:
            print(span + "  implied accel bias [mg] " + " ".join(f"{v / G * 1000:+6.1f}" for v in bias))
        else:
            print(span + "  NOT corrected (no still moment after it), drifts freely")

    pos_f = pose.at_frames(sol, pos) + pose.qrot(sol.qf, np.broadcast_to(np.array(args.offset), (len(sol.qf), 3)))
    out = args.out or args.recording.rsplit(".", 1)[0] + ".track.csv"
    pose.write_frames(out, "nav-track v1", sol, pos_f, [
        f"recording={args.recording}",
        "q rotates body to world. body: +X ToF look direction, +Z grid up. "
        "world: +Z up, +X initial look direction, origin = IMU position at the start",
        "tof_* is the tracked ToF position (double-integrated accel, zero velocity at still moments)",
        f"offset_m={','.join(str(v) for v in args.offset)}",
        f"tof_shift_ms={args.tof_shift_ms}",
    ])

    path_len = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
    print(f"\npath length {path_len:.2f} m, farthest from start {np.linalg.norm(pos, axis=1).max():.2f} m, "
          f"end {np.linalg.norm(pos[-1]):.2f} m from start "
          f"({', '.join(f'{v:+.2f}' for v in pos[-1])}), top speed {np.linalg.norm(vel, axis=1).max():.2f} m/s")

    if args.no_plot and not args.save:
        return
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1])
    axs = [fig.add_subplot(gs[k, 0]) for k in range(3)]
    for ax in axs:
        for s, e in sol.segs:
            ax.axvspan(rec.t[s], rec.t[e - 1], color="#1baf7a", alpha=0.15, lw=0)
        ax.grid(True, ls=":", alpha=0.5)
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    for k, name in enumerate("XYZ"):
        axs[0].plot(rec.t, acc_world[:, k], color=cols[k], lw=0.6, label=name)
        axs[1].plot(rec.t, vel[:, k], color=cols[k], label=name)
        axs[2].plot(rec.t, pos[:, k], color=cols[k], label=name)
    axs[0].set_ylabel("world accel [m/s²]")
    axs[0].set_title(f"{args.recording}: tracked motion (green = still moments)")
    axs[1].set_ylabel("velocity [m/s]")
    axs[2].set_ylabel("position [m]")
    axs[2].set_xlabel("t [s]")
    for ax in axs:
        ax.legend(loc="upper right")
    top = fig.add_subplot(gs[:, 1])
    top.plot(pos[:, 0], pos[:, 1], color="0.4", lw=1)
    top.plot(pos[0, 0], pos[0, 1], "o", color="#1baf7a", label="start")
    top.plot(pos[-1, 0], pos[-1, 1], "s", color="#e74c3c", label="end")
    top.set_aspect("equal", adjustable="datalim")
    top.set_xlabel("X [m] (initial look)")
    top.set_ylabel("Y [m] (left)")
    top.set_title("path seen from above")
    top.grid(True, ls=":", alpha=0.5)
    top.legend()
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()
