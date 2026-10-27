"""Shared host-side link to the firmware: wire format, framing, CRC check, request loop.

Wire format is documented in src/main.rs. One packet per request byte.
"""

import struct
import sys
import time
import zlib
from dataclasses import dataclass

import numpy as np
import serial

PORT = "/dev/ttyACM0"
BAUDRATE = 921600
REQ_BYTE = b"\x23"  # 0x23 ('#')

# --- Wire Format Definitions ---
MAGIC = b"NAV2"
HEADER_FMT = "<4sIQQIIHHHHHHH3hHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 56 bytes
ZONES_SIZE = 64 * 4 * 2                    # 512 bytes (64 zones * 4 targets * 2 B)
SAMPLE_SIZE = 14                           # 7 * i16 (ax, ay, az, gx, gy, gz, temp)
CRC_SIZE = 4
MAX_N_IMU = 128                            # sanity limit, firmware MAX_BATCH is 64

FLAG_IMU_GAP_BEFORE = 1 << 0  # data lost right before the first IMU sample
FLAG_IMU_CLIPPED = 1 << 1     # an axis hit full scale in this batch
FLAG_IMU_BACKLOG = 1 << 2     # more samples waiting, next packet continues
FLAG_IMU_BIAS_VALID = 1 << 3  # gyro_bias was measured while still

REQUEST_TIMEOUT_S = 0.5  # no packet start seen for this long -> ask again
PACKET_TIMEOUT_S = 1.0   # packet started but never finished -> drop it and ask again


def temp_to_c(raw):
    """MPU-6050 die temperature from raw LSB."""
    return raw / 340.0 + 36.53


@dataclass
class Packet:
    host_t: float  # time.monotonic() when the packet was fully received
    vl53_seq: int
    vl53_t_us: int
    imu_t_us: int  # MCU time of the LAST imu sample
    imu_seq0: int  # absolute index of the first imu sample
    imu_lost: int
    n_imu: int
    flags: int
    imu_rate_hz: int
    imu_resets: int
    imu_errors: int
    accel_lsb_per_g: int
    gyro_lsb_per_dps_x10: int
    gyro_bias: tuple  # raw LSB, measured at boot
    imu_misaligned: int  # cumulative FIFO resets after a frame failed the alignment check
    zones: np.ndarray  # (64, 4) i16 distances in mm, -1 = invalid
    imu: np.ndarray    # (n_imu, 7) i16 raw: ax ay az gx gy gz temp

    @property
    def accel_scale(self):
        return float(self.accel_lsb_per_g) if self.accel_lsb_per_g > 0 else 16384.0

    @property
    def gyro_scale(self):
        return self.gyro_lsb_per_dps_x10 / 10.0 if self.gyro_lsb_per_dps_x10 > 0 else 65.5

    @property
    def accel_g(self):
        return self.imu[:, 0:3] / self.accel_scale

    @property
    def gyro_dps(self):
        """Gyro rates with the boot bias removed."""
        return (self.imu[:, 3:6] - np.array(self.gyro_bias)) / self.gyro_scale

    @property
    def temp_c(self):
        return temp_to_c(self.imu[:, 6].astype(np.float64))


def parse_packet(packet: bytes, host_t: float) -> Packet:
    """Parse one complete, CRC-checked packet."""
    h = struct.unpack(HEADER_FMT, packet[:HEADER_SIZE])
    n_imu = h[6]
    zones = np.frombuffer(packet, dtype="<i2", count=64 * 4, offset=HEADER_SIZE).reshape((64, 4))
    imu = np.frombuffer(
        packet, dtype="<i2", count=n_imu * 7, offset=HEADER_SIZE + ZONES_SIZE
    ).reshape((n_imu, 7))
    return Packet(
        host_t=host_t,
        vl53_seq=h[1],
        vl53_t_us=h[2],
        imu_t_us=h[3],
        imu_seq0=h[4],
        imu_lost=h[5],
        n_imu=n_imu,
        flags=h[7],
        imu_rate_hz=h[8],
        imu_resets=h[9],
        imu_errors=h[10],
        accel_lsb_per_g=h[11],
        gyro_lsb_per_dps_x10=h[12],
        gyro_bias=(h[13], h[14], h[15]),
        imu_misaligned=h[16],
        zones=zones,
        imu=imu,
    )


def open_port(port=PORT):
    return serial.Serial(port, BAUDRATE, timeout=0.1)


def _request(ser):
    ser.write(REQ_BYTE)
    ser.flush()
    return time.monotonic()


def read_packets(ser):
    """Generator: runs the request/response loop and yields every valid Packet.

    Handles magic resync, timeouts and CRC errors. The next packet is requested before
    yielding, so the MCU is already working on it while the caller processes this one.
    """
    buf = bytearray()
    ser.reset_input_buffer()
    last_req_time = _request(ser)

    while True:
        waiting = ser.in_waiting
        chunk = ser.read(waiting if waiting > 0 else 1)
        if chunk:
            buf.extend(chunk)

        # Search for packet magic sync
        idx = buf.find(MAGIC)
        if idx < 0:
            if len(buf) > 3:
                del buf[:-3]
            if time.monotonic() - last_req_time > REQUEST_TIMEOUT_S:
                last_req_time = _request(ser)
            continue

        if idx > 0:
            del buf[:idx]

        # Wait for at least the full header
        if len(buf) < HEADER_SIZE:
            continue

        n_imu = struct.unpack(HEADER_FMT, bytes(buf[:HEADER_SIZE]))[6]
        if n_imu > MAX_N_IMU:
            del buf[:4]
            continue

        total_packet_len = HEADER_SIZE + ZONES_SIZE + n_imu * SAMPLE_SIZE + CRC_SIZE

        if len(buf) < total_packet_len:
            if time.monotonic() - last_req_time > PACKET_TIMEOUT_S:
                del buf[:4]
                last_req_time = _request(ser)
            continue

        packet = bytes(buf[:total_packet_len])
        del buf[:total_packet_len]
        host_t = time.monotonic()

        # Validate CRC-32 (IEEE / zlib)
        pkt_crc = struct.unpack("<I", packet[-CRC_SIZE:])[0]
        calc_crc = zlib.crc32(packet[:-CRC_SIZE]) & 0xFFFFFFFF
        if calc_crc != pkt_crc:
            print(f"[CRC Error] expected 0x{pkt_crc:08x}, got 0x{calc_crc:08x}", file=sys.stderr)
            last_req_time = _request(ser)
            continue

        last_req_time = _request(ser)
        yield parse_packet(packet, host_t)
