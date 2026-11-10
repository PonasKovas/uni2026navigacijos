"""Offline analysis of recordings made with record.py.

    python analyze.py info walk1.navrec              # what is in the file
    python analyze.py tof walk1.navrec               # ToF per-channel statistics (valid, stuck, ...)
    python analyze.py tof walk1.navrec --zone 10     # also dump one zone frame by frame

Own scripts can load a recording with navlink.load_recording(path), which gives the same
navlink.Packet objects as the live link.
"""

import argparse
import json

import numpy as np

import navlink


# ============================================================ info

def cmd_info(args):
    meta, pkts, marks = navlink.load_recording(args.file)
    print(json.dumps(meta, indent=2))
    if not pkts:
        print("no packets")
        return
    t0, t1 = pkts[0].host_t, pkts[-1].host_t
    seqs = np.array([p.vl53_seq for p in pkts], dtype=np.int64)
    n_imu = sum(p.n_imu for p in pkts)
    print(f"frames      {len(pkts)} over {t1 - t0:.1f} s ({(len(pkts) - 1) / max(t1 - t0, 1e-9):.2f}/s)")
    print(f"vl53_seq    {seqs[0]} .. {seqs[-1]}, jumps {(np.diff(seqs) != 1).sum()}")
    print(f"imu samples {n_imu}")
    last = pkts[-1]
    print(f"imu counters at end: lost {last.imu_lost}, resets {last.imu_resets}, "
          f"errors {last.imu_errors}, misaligned {last.imu_misaligned}")
    for t, label in marks:
        print(f"mark {t:8.2f} s  {label}")


# ============================================================ tof

def _runs(x):
    """Lengths of runs of equal consecutive values in a 1D array."""
    if len(x) == 0:
        return np.array([], dtype=int)
    edges = np.flatnonzero(np.diff(x) != 0) + 1
    return np.diff(np.concatenate(([0], edges, [len(x)])))


def cmd_tof(args):
    _, pkts, _ = navlink.load_recording(args.file)
    if not pkts:
        print("no packets")
        return
    z = np.stack([p.zones for p in pkts])  # (frames, 64, 4)
    valid = z >= 0
    print(f"{len(z)} frames\n")

    print("channel  valid%   ch valid but previous ch invalid   distance mm (median, min..max)")
    for ch in range(4):
        v = valid[:, :, ch]
        orphan = (v & ~valid[:, :, ch - 1]).sum() if ch > 0 else 0
        d = z[:, :, ch][v]
        dist = f"{np.median(d):6.0f}  {d.min()}..{d.max()}" if d.size else "-"
        print(f"   {ch}     {100 * v.mean():6.2f}   {orphan:8d} ({100 * orphan / max(v.sum(), 1):5.1f}% of valid)"
              f"           {dist}")

    # A real reading jitters by a few mm every frame. The same exact value over many frames
    # while the sensor moves is a sign of stale data.
    print(f"\nstuck: a zone holding the exact same valid value for >= {args.stuck} frames in a row")
    print("channel  zones with a stuck run   longest run (frames)   frames inside stuck runs")
    for ch in range(4):
        zones_hit = longest = inside = 0
        for zone in range(64):
            col = z[:, zone, ch]
            lengths = _runs(col)
            starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
            stuck = (col[starts] >= 0) & (lengths >= args.stuck)
            if stuck.any():
                zones_hit += 1
                longest = max(longest, int(lengths[stuck].max()))
                inside += int(lengths[stuck].sum())
        print(f"   {ch}     {zones_hit:6d}                 {longest:8d}               {inside:8d}")

    if args.zone is not None:
        print(f"\nzone {args.zone} (row {args.zone // 8}, col {args.zone % 8}), every frame:")
        print("frame  host_t   ch0    ch1    ch2    ch3")
        for i, p in enumerate(pkts):
            print(f"{i:5d} {p.host_t:7.2f}  " + "  ".join(f"{v:5d}" for v in p.zones[args.zone]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("info", help="summary of a recording")
    i.add_argument("file")
    i.set_defaults(func=cmd_info)

    t = sub.add_parser("tof", help="ToF per-channel statistics")
    t.add_argument("file")
    t.add_argument("--stuck", type=int, default=15, help="frames, run length that counts as stuck (15 = 1 s)")
    t.add_argument("--zone", type=int, help="also dump this zone (0..63) frame by frame")
    t.set_defaults(func=cmd_tof)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
