"""Record everything the MCU sends to a file, for offline work with analyze.py.

    python record.py walk1.navrec          # Ctrl+C to stop. Enter = mark (type a label first if you want)
    python record.py walk1.navrec -d 60    # stop by itself after 60 s

Every CRC-checked packet is saved byte for byte (header, ToF zones, IMU batch, CRC) with the host
receive time. File format: see navlink.py.
"""

import argparse
import datetime
import json
import queue
import struct
import sys
import threading
import time

import navlink

STATUS_PERIOD_S = 0.5


def _stdin_marks(q):
    """Each line typed on stdin becomes a mark, timed when Enter is pressed."""
    for line in sys.stdin:
        q.put((time.monotonic(), line.strip()))


def _write_record(f, kind, host_t, payload):
    f.write(struct.pack(navlink.REC_HEADER_FMT, kind, host_t, len(payload)))
    f.write(payload)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--port", default=navlink.PORT)
    p.add_argument("-d", "--duration", type=float, default=0, help="stop after this many seconds (0 = Ctrl+C)")
    args = p.parse_args()

    marks = queue.Queue()
    threading.Thread(target=_stdin_marks, args=(marks,), daemon=True).start()

    n_packets = n_imu = n_marks = n_bytes = 0
    last_status = 0.0

    with navlink.open_port(args.port) as ser, open(args.out, "wb") as f:
        t_ref = time.monotonic()
        f.write(navlink.REC_MAGIC)
        meta = {
            "start_wall": datetime.datetime.now().isoformat(timespec="seconds"),
            "port": args.port,
            "wire_magic": navlink.MAGIC.decode(),
        }
        f.write(json.dumps(meta).encode() + b"\n")
        print("recording, Ctrl+C to stop. Enter = mark (type a label before Enter if you want)",
              file=sys.stderr)
        try:
            for pkt in navlink.read_packets(ser):
                while not marks.empty():
                    t, label = marks.get()
                    _write_record(f, b"M", t - t_ref, label.encode())
                    n_marks += 1
                    print(f"\nmark {n_marks} at {t - t_ref:.2f} s {label!r}", file=sys.stderr)

                _write_record(f, b"P", pkt.host_t - t_ref, pkt.raw)
                n_packets += 1
                n_imu += pkt.n_imu
                n_bytes += len(pkt.raw)

                now = time.monotonic()
                if now - last_status > STATUS_PERIOD_S:
                    last_status = now
                    el = now - t_ref
                    print(f"\r{el:7.1f} s  {n_packets} frames ({n_packets / el:.1f}/s)  {n_imu} imu  "
                          f"{n_bytes / 1e6:.1f} MB  marks {n_marks}   ", end="", file=sys.stderr)

                if args.duration and now - t_ref >= args.duration:
                    break
        except KeyboardInterrupt:
            pass
    print(f"\nsaved {n_packets} frames, {n_imu} imu samples, {n_marks} marks to {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
