"""Visual check of a pose.py result: replays the rotation in real time.

    python visualize_pose.py test3.pose.csv            # real time
    python visualize_pose.py test3.pose.csv --speed 2  # twice as fast

Keys: space = pause, left/right = -/+ 5 s, q = quit.
Left: the sensor in the world (Z up, X = initial look direction). Red = look direction (+X),
green = +Y, blue = grid up (+Z), gray = trail of the ToF position. Right: yaw/pitch/roll with
the current time.
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

    d = np.genfromtxt(args.pose_csv, delimiter=",", comments="#")
    t = d[:, 2] - d[0, 2]
    q = d[:, 5:9]
    pos = d[:, 9:12]
    yaw, pitch, roll = d[:, 12], d[:, 13], d[:, 14]
    still = d[:, 3].astype(bool)

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    r = max(0.15, 1.6 * np.abs(pos).max())
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X [m] (initial look)")
    ax.set_ylabel("Y [m] (left)")
    ax.set_zlabel("Z [m] (up)")
    ax.plot([0], [0], [0], "ko", ms=5)
    ax.plot([-r, r], [0, 0], [0, 0], color="0.85", lw=0.8)
    ax.plot([0, 0], [-r, r], [0, 0], color="0.85", lw=0.8)
    (arm,) = ax.plot([], [], [], color="0.3", lw=2)
    (trail,) = ax.plot([], [], [], color="0.6", lw=1)
    axes_lines = [ax.plot([], [], [], color=c, lw=3)[0] for c in ("#e74c3c", "#2ecc71", "#3498db")]
    (ray,) = ax.plot([], [], [], color="#e74c3c", lw=1, ls="--")
    title = ax.set_title("")

    axs = [fig.add_subplot(3, 2, k) for k in (2, 4, 6)]
    cursors = []
    for a, y, name in zip(axs, (yaw, pitch, roll), ("yaw [°]", "pitch [°] (up +)", "roll [°]")):
        a.plot(t, y, lw=1)
        a.fill_between(t, y.min(), y.max(), where=still, color="#1baf7a", alpha=0.15, lw=0)
        a.set_ylabel(name)
        a.grid(True, ls=":", alpha=0.5)
        cursors.append(a.axvline(0, color="k", lw=1))
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
        arm.set_data([0, p0[0]], [0, p0[1]])
        arm.set_3d_properties([0, p0[2]])
        k0 = max(0, i - 15 * 10)  # last ~10 s
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
