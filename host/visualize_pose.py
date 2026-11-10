"""Visual check of a pose.py or track.py result: replays the motion in real time.

    python visualize_pose.py test3.pose.csv            # real time
    python visualize_pose.py test3.pose.csv --speed 2  # twice as fast
    python visualize_pose.py test10.track.csv          # hand-held, tracked position

Keys: space = pause, left/right = -/+ 5 s, q = quit.
Left: the sensor in the world (Z up, X = initial look direction). Red = look direction (+X),
green = +Y, blue = grid up (+Z). Pose files: black arm from the pivot, gray = last 10 s of the
ToF position. Track files: gray = the whole path so far. Right: yaw/pitch/roll (and position
for track files) with the current time.
"""

import argparse
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def qrot(q, v):
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return R @ v


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pose_csv")
    p.add_argument("--speed", type=float, default=1.0)
    args = p.parse_args()

    with open(args.pose_csv) as f:
        track = f.readline().startswith("# nav-track")  # pose.py writes "# nav-pose"
    d = np.genfromtxt(args.pose_csv, delimiter=",", comments="#")
    t = d[:, 2] - d[0, 2]
    q = d[:, 5:9]
    pos = d[:, 9:12]
    yaw, pitch, roll = d[:, 12], d[:, 13], d[:, 14]
    still = d[:, 3].astype(bool)

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    if track:
        # cube around the whole path
        c = (pos.min(axis=0) + pos.max(axis=0)) / 2
        r = max(0.3, 0.6 * np.ptp(pos, axis=0).max())
    else:
        c = np.zeros(3)
        r = max(0.15, 1.6 * np.abs(pos).max())
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X [m] (initial look)")
    ax.set_ylabel("Y [m] (left)")
    ax.set_zlabel("Z [m] (up)")
    ax.plot([0], [0], [0], "ko", ms=5)  # pivot, or the start for track files
    if not track:
        ax.plot([-r, r], [0, 0], [0, 0], color="0.85", lw=0.8)
        ax.plot([0, 0], [-r, r], [0, 0], color="0.85", lw=0.8)
    (arm,) = ax.plot([], [], [], color="0.3", lw=2)
    (trail,) = ax.plot([], [], [], color="0.6", lw=1)
    axes_lines = [ax.plot([], [], [], color=c, lw=3)[0] for c in ("#e74c3c", "#2ecc71", "#3498db")]
    (ray,) = ax.plot([], [], [], color="#e74c3c", lw=1, ls="--")
    title = ax.set_title("")

    rows = 4 if track else 3
    axs = [fig.add_subplot(rows, 2, 2 * k + 2) for k in range(rows)]
    cursors = []
    for a, y, name in zip(axs, (yaw, pitch, roll), ("yaw [°]", "pitch [°] (up +)", "roll [°]")):
        a.plot(t, y, lw=1)
        a.fill_between(t, y.min(), y.max(), where=still, color="#1baf7a", alpha=0.15, lw=0)
        a.set_ylabel(name)
        a.grid(True, ls=":", alpha=0.5)
        cursors.append(a.axvline(0, color="k", lw=1))
    if track:
        for k, (name, col) in enumerate(zip("XYZ", ("#e74c3c", "#2ecc71", "#3498db"))):
            axs[3].plot(t, pos[:, k], color=col, lw=1, label=name)
        axs[3].set_ylabel("position [m]")
        axs[3].legend(loc="upper right", fontsize=8)
        axs[3].grid(True, ls=":", alpha=0.5)
        cursors.append(axs[3].axvline(0, color="k", lw=1))
    axs[-1].set_xlabel("t [s] (green = still)")

    state = {"t0": time.monotonic(), "offset": 0.0, "paused": False, "pause_t": 0.0}

    def now():
        if state["paused"]:
            return state["pause_t"]
        return (time.monotonic() - state["t0"]) * args.speed + state["offset"]

    def on_key(e):
        cur = now()
        if e.key == " ":
            state["paused"] = not state["paused"]
            state["pause_t"] = cur
        elif e.key in ("left", "right"):
            cur = float(np.clip(cur + (5 if e.key == "right" else -5), 0, t[-1]))
            state["pause_t"] = cur
        else:
            return
        state["t0"], state["offset"] = time.monotonic(), cur

    fig.canvas.mpl_connect("key_press_event", on_key)
    arrow = 0.4 * r

    def update(_):
        cur = now()
        if cur > t[-1]:  # loop
            state["t0"], state["offset"], cur = time.monotonic(), 0.0, 0.0
        i = int(np.clip(np.searchsorted(t, cur), 0, len(t) - 1))
        p0 = pos[i]
        if not track:
            arm.set_data([0, p0[0]], [0, p0[1]])
            arm.set_3d_properties([0, p0[2]])
        k0 = 0 if track else max(0, i - 15 * 10)  # whole path, or the last ~10 s
        trail.set_data(pos[k0:i + 1, 0], pos[k0:i + 1, 1])
        trail.set_3d_properties(pos[k0:i + 1, 2])
        for line, axis in zip(axes_lines, np.eye(3)):
            tip = p0 + arrow * qrot(q[i], axis)
            line.set_data([p0[0], tip[0]], [p0[1], tip[1]])
            line.set_3d_properties([p0[2], tip[2]])
        far = p0 + 2.5 * r * qrot(q[i], np.array([1.0, 0, 0]))
        ray.set_data([p0[0], far[0]], [p0[1], far[1]])
        ray.set_3d_properties([p0[2], far[2]])
        title.set_text(f"t {t[i]:6.1f} s   yaw {yaw[i]:+7.1f}°  pitch {pitch[i]:+6.1f}°  roll {roll[i]:+5.1f}°"
                       + ("   PAUSED" if state["paused"] else ""))
        for c in cursors:
            c.set_xdata([t[i], t[i]])
        return []

    ani = animation.FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)  # noqa: F841
    fig.suptitle(f"{args.pose_csv}   (space = pause, ←/→ = ∓5 s, q = quit)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


if __name__ == "__main__":
    main()
