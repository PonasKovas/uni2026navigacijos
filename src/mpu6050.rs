//! MPU-6050 driver: hardware FIFO @ 200 Hz -> drained by a task -> ring buffer -> `take_batch()`.
//!
//! The FIFO is what makes this reliable: vl53l5cx's blocking I2C readout stalls the executor
//! for tens of ms, but the MPU keeps buffering (73 frames = ~365 ms at 200 Hz).
//! Raw sensor axes, raw LSB values. Scale factors are exported for the packet header.
//! Temperature: °C = raw / 340 + 36.53 (die temperature, not ambient).

use core::cell::RefCell;

use defmt::{info, warn};
use embassy_stm32::{
    exti::ExtiInput,
    i2c::{self, I2c, Master},
    mode::Async,
};
use embassy_sync::blocking_mutex::{Mutex, raw::CriticalSectionRawMutex};
use embassy_time::{Duration, Instant, Timer, with_timeout};
use zerocopy::{Immutable, IntoBytes, KnownLayout};

// ---------------------------------------------------------------- device constants
const ADDR: u8 = 0x68; // 7-bit address (AD0 low). embassy wants the 7-bit form, no R/W bit.
const WHO_AM_I_VALUE: u8 = 0x68;

const REG_SMPLRT_DIV: u8 = 0x19; // 25
const REG_CONFIG: u8 = 0x1a; // 26
const REG_GYRO_CONFIG: u8 = 0x1b; // 27
const REG_ACCEL_CONFIG: u8 = 0x1c; // 28
const REG_FIFO_EN: u8 = 0x23; // 35
const REG_INT_PIN_CFG: u8 = 0x37; // 55
const REG_INT_ENABLE: u8 = 0x38; // 56
const REG_SIGNAL_PATH_RESET: u8 = 0x68; // 104
const REG_USER_CTRL: u8 = 0x6a; // 106
const REG_PWR_MGMT_1: u8 = 0x6b; // 107
const REG_PWR_MGMT_2: u8 = 0x6c; // 108
const REG_FIFO_COUNT_H: u8 = 0x72; // 114
const REG_FIFO_R_W: u8 = 0x74; // 116
const REG_WHO_AM_I: u8 = 0x75; // 117

const PWR1_DEVICE_RESET: u8 = 1 << 7;
const PWR1_CLKSEL_PLL_XGYRO: u8 = 1;
const SIGNAL_PATH_RESET_ALL: u8 = 0b111;
const USER_CTRL_FIFO_EN: u8 = 1 << 6;
const USER_CTRL_FIFO_RESET: u8 = 1 << 2;
const FIFO_EN_TEMP_ACCEL_GYRO: u8 = (1 << 7) | (1 << 6) | (1 << 5) | (1 << 4) | (1 << 3); // TEMP XG YG ZG ACCEL
const INT_ENABLE_DATA_RDY: u8 = 1;

// ---------------------------------------------------------------- configuration
const DLPF_CFG: u8 = 3; // accel 44 Hz / gyro 42 Hz, ~4.8 ms delay, gyro output rate 1 kHz
const SMPLRT_DIV: u8 = 4; // 1 kHz / (1 + 4) = 200 Hz
const GYRO_FS_SEL: u8 = 1; // 0:±250 1:±500 2:±1000 3:±2000 dps
const ACCEL_AFS_SEL: u8 = 0; // 0:±2g 1:±4g 2:±8g 3:±16g

const _: () = assert!(DLPF_CFG >= 1 && DLPF_CFG <= 6); // 0/7 would make the gyro rate 8 kHz
const _: () = assert!(GYRO_FS_SEL <= 3 && ACCEL_AFS_SEL <= 3);

pub const SAMPLE_RATE_HZ: u16 = 1000 / (SMPLRT_DIV as u16 + 1);
pub const ACCEL_LSB_PER_G: u16 = 16384 >> ACCEL_AFS_SEL;
/// LSB per dps, times 10 (65.5 LSB/dps -> 655)
pub const GYRO_LSB_PER_DPS_X10: u16 = match GYRO_FS_SEL {
    0 => 1310,
    1 => 655,
    2 => 328,
    _ => 164,
};

// FIFO is filled in register order: ax ay az temp gx gy gz, big-endian i16 each
const FRAME_BYTES: usize = 14;
const FIFO_SIZE: usize = 1024;
const FIFO_INTACT_MAX: usize = FIFO_SIZE / FRAME_BYTES * FRAME_BYTES; // 1022; above this we can't trust alignment
const READ_FRAMES: usize = 16; // frames per I2C burst

const RING_LEN: usize = 256; // ~1.28 s of samples if the host stalls
pub const MAX_BATCH: usize = 64;

const INT_CHECK_TIMEOUT: Duration = Duration::from_millis(100); // INT fires every 5 ms at 200 Hz
const WAKE_TIMEOUT: Duration = Duration::from_millis(20); // INT silent for this long = something is wrong

// boot-time gyro bias measurement (device must be still)
const CAL_SAMPLES: usize = SAMPLE_RATE_HZ as usize; // 1 s
const CAL_SKIP: usize = 20;
const CAL_ATTEMPTS: usize = 3;
const CAL_MAX_POLLS: usize = 100;
const CAL_MAX_RANGE_DPS_X10: i32 = 15; // peak-to-peak 1.5 dps allowed while "still"

// ---------------------------------------------------------------- public types
#[derive(Clone, Copy, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct Sample {
    pub accel: [i16; 3],
    pub gyro: [i16; 3],
    pub temp: i16,
}

impl Sample {
    pub const ZERO: Sample = Sample {
        accel: [0; 3],
        gyro: [0; 3],
        temp: 0,
    };
}

#[derive(defmt::Format)]
pub enum InitError {
    I2c(i2c::Error),
    WrongDevice(u8),
    IntDead, // no data-ready edge: INT wire missing or on the wrong pin
}

impl From<i2c::Error> for InitError {
    fn from(e: i2c::Error) -> Self {
        InitError::I2c(e)
    }
}

pub struct Batch {
    /// number of samples written to the output slice
    pub n: usize,
    /// absolute index of the first sample (counts successfully received samples since boot)
    pub seq0: u32,
    /// estimated MCU time (µs) of the LAST sample in the batch
    pub t_us: u64,
    /// stream is discontinuous immediately before the first sample (lost data)
    pub gap_before: bool,
    /// some sample hit full scale
    pub clipped: bool,
    /// more samples are waiting in the ring (batch truncated)
    pub backlog: bool,
    /// cumulative samples dropped because the ring was full
    pub lost: u32,
    /// cumulative FIFO overflow resets
    pub resets: u16,
    /// cumulative I2C errors
    pub errors: u16,
    /// gyro bias measured at boot, raw LSB (zeros if `calib_ok` is false)
    pub gyro_bias: [i16; 3],
    pub calib_ok: bool,
}

// ---------------------------------------------------------------- shared ring
const ENTRY_GAP: u8 = 1 << 0;
const ENTRY_CLIPPED: u8 = 1 << 1;

#[derive(Clone, Copy)]
struct Entry {
    sample: Sample,
    flags: u8,
}

impl Entry {
    const ZERO: Entry = Entry {
        sample: Sample::ZERO,
        flags: 0,
    };
}

struct Shared {
    ring: [Entry; RING_LEN],
    head: usize,
    len: usize,
    next_seq: u32,
    pending_gap: bool,
    newest_t_us: u64,
    lost: u32,
    resets: u16,
    errors: u16,
    gyro_bias: [i16; 3],
    calib_ok: bool,
}

impl Shared {
    const fn new() -> Self {
        Self {
            ring: [Entry::ZERO; RING_LEN],
            head: 0,
            len: 0,
            next_seq: 0,
            pending_gap: false,
            newest_t_us: 0,
            lost: 0,
            resets: 0,
            errors: 0,
            gyro_bias: [0; 3],
            calib_ok: false,
        }
    }

    fn push(&mut self, sample: Sample, t_us: u64) {
        let mut flags = 0;
        if self.pending_gap {
            flags |= ENTRY_GAP;
            self.pending_gap = false;
        }
        if is_clipped(&sample) {
            flags |= ENTRY_CLIPPED;
        }
        if self.len == RING_LEN {
            // ring full: drop the oldest, its successor now follows a hole
            self.head = (self.head + 1) % RING_LEN;
            self.len -= 1;
            self.lost = self.lost.wrapping_add(1);
            let h = self.head;
            self.ring[h].flags |= ENTRY_GAP;
        }
        let tail = (self.head + self.len) % RING_LEN;
        self.ring[tail] = Entry { sample, flags };
        self.len += 1;
        self.next_seq = self.next_seq.wrapping_add(1);
        self.newest_t_us = t_us;
    }
}

static SHARED: Mutex<CriticalSectionRawMutex, RefCell<Shared>> =
    Mutex::new(RefCell::new(Shared::new()));

fn is_clipped(s: &Sample) -> bool {
    s.accel
        .iter()
        .chain(s.gyro.iter())
        .any(|&v| v >= i16::MAX - 8 || v <= i16::MIN + 8)
}

/// Move up to `out.len()` samples from the ring into `out`. Never spans a discontinuity:
/// a batch ends right before a gap, the next batch starts with `gap_before = true`.
pub fn take_batch(out: &mut [Sample]) -> Batch {
    SHARED.lock(|cell| {
        let mut guard = cell.borrow_mut();
        let s = &mut *guard;

        let seq0 = s.next_seq.wrapping_sub(s.len as u32);
        let mut n = 0;
        let mut gap_before = false;
        let mut clipped = false;

        while n < out.len() && s.len > 0 {
            let e = s.ring[s.head];
            if n > 0 && e.flags & ENTRY_GAP != 0 {
                break;
            }
            if n == 0 {
                gap_before = e.flags & ENTRY_GAP != 0;
            }
            clipped |= e.flags & ENTRY_CLIPPED != 0;
            out[n] = e.sample;
            s.head = (s.head + 1) % RING_LEN;
            s.len -= 1;
            n += 1;
        }

        let t_us = s
            .newest_t_us
            .saturating_sub(s.len as u64 * 1_000_000 / SAMPLE_RATE_HZ as u64);

        Batch {
            n,
            seq0,
            t_us,
            gap_before,
            clipped,
            backlog: s.len > 0,
            lost: s.lost,
            resets: s.resets,
            errors: s.errors,
            gyro_bias: s.gyro_bias,
            calib_ok: s.calib_ok,
        }
    })
}

// ---------------------------------------------------------------- driver
pub struct Mpu6050 {
    i2c: I2c<'static, Async, Master>,
    int: ExtiInput<'static, Async>,
}

enum Drain {
    Drained,
    Overflow,
}

fn parse_frame(b: &[u8]) -> Sample {
    let v = |i: usize| i16::from_be_bytes([b[2 * i], b[2 * i + 1]]);
    Sample {
        accel: [v(0), v(1), v(2)],
        gyro: [v(4), v(5), v(6)],
        temp: v(3),
    }
}

impl Mpu6050 {
    pub async fn init(
        i2c: I2c<'static, Async, Master>,
        int: ExtiInput<'static, Async>,
    ) -> Result<Self, InitError> {
        let mut this = Self { i2c, int };

        Timer::after_millis(100).await; // power-up time before register access

        let who = this.read_reg(REG_WHO_AM_I).await?;
        if who != WHO_AM_I_VALUE {
            return Err(InitError::WrongDevice(who));
        }

        this.write_reg(REG_PWR_MGMT_1, PWR1_DEVICE_RESET).await?;
        Timer::after_millis(100).await;
        this.write_reg(REG_SIGNAL_PATH_RESET, SIGNAL_PATH_RESET_ALL)
            .await?;
        Timer::after_millis(100).await;
        // wake up (SLEEP=0), clock from gyro X PLL (much better than the internal 8 MHz)
        this.write_reg(REG_PWR_MGMT_1, PWR1_CLKSEL_PLL_XGYRO)
            .await?;
        Timer::after_millis(100).await;

        this.write_reg(REG_PWR_MGMT_2, 0).await?;
        this.write_reg(REG_SMPLRT_DIV, SMPLRT_DIV).await?;
        this.write_reg(REG_CONFIG, DLPF_CFG).await?;
        this.write_reg(REG_GYRO_CONFIG, GYRO_FS_SEL << 3).await?;
        this.write_reg(REG_ACCEL_CONFIG, ACCEL_AFS_SEL << 3).await?;
        this.write_reg(REG_INT_PIN_CFG, 0).await?; // active high, push-pull, 50 µs pulse
        this.write_reg(REG_INT_ENABLE, INT_ENABLE_DATA_RDY).await?;
        this.write_reg(REG_FIFO_EN, FIFO_EN_TEMP_ACCEL_GYRO).await?;
        this.reset_fifo().await?; // also starts the FIFO

        // INT must toggle at the sample rate. If it doesn't, fail now instead of limping along.
        with_timeout(INT_CHECK_TIMEOUT, this.int.wait_for_rising_edge())
            .await
            .map_err(|_| InitError::IntDead)?;

        let mut bias = [0i16; 3];
        let mut ok = false;
        for attempt in 0..CAL_ATTEMPTS {
            match this.measure_gyro_bias().await? {
                Some(b) => {
                    bias = b;
                    ok = true;
                    break;
                }
                None => warn!(
                    "MPU6050 gyro calibration: not still, attempt {}/{}",
                    attempt + 1,
                    CAL_ATTEMPTS
                ),
            }
        }

        SHARED.lock(|c| {
            let mut s = c.borrow_mut();
            s.gyro_bias = bias;
            s.calib_ok = ok;
        });

        info!(
            "MPU6050 ready: {} Hz, gyro bias {} LSB, calib_ok={}",
            SAMPLE_RATE_HZ, bias, ok
        );

        Ok(this)
    }

    async fn write_reg(&mut self, reg: u8, val: u8) -> Result<(), i2c::Error> {
        self.i2c.write(ADDR, &[reg, val]).await
    }

    async fn read_reg(&mut self, reg: u8) -> Result<u8, i2c::Error> {
        let mut b = [0u8; 1];
        self.i2c.write_read(ADDR, &[reg], &mut b).await?;
        Ok(b[0])
    }

    /// disable -> reset -> enable (FIFO_RESET only works while FIFO_EN = 0)
    async fn reset_fifo(&mut self) -> Result<(), i2c::Error> {
        self.write_reg(REG_USER_CTRL, 0).await?;
        self.write_reg(REG_USER_CTRL, USER_CTRL_FIFO_RESET).await?;
        self.write_reg(REG_USER_CTRL, USER_CTRL_FIFO_EN).await
    }

    /// Pops every complete frame currently in the FIFO into `sink`.
    async fn drain(&mut self, mut sink: impl FnMut(Sample)) -> Result<Drain, i2c::Error> {
        // FIFO_COUNT must be read high-then-low: one 2-byte burst starting at H.
        let mut cnt = [0u8; 2];
        self.i2c
            .write_read(ADDR, &[REG_FIFO_COUNT_H], &mut cnt)
            .await?;
        let count = u16::from_be_bytes(cnt) as usize;

        if count > FIFO_INTACT_MAX {
            // overwritten oldest bytes -> frame alignment unknown, start over
            self.reset_fifo().await?;
            return Ok(Drain::Overflow);
        }

        // count can be caught mid-frame-write: only take whole frames
        let total = count / FRAME_BYTES;
        let mut frames = total;
        let mut buf = [0u8; READ_FRAMES * FRAME_BYTES];
        while frames > 0 {
            let n = frames.min(READ_FRAMES);
            let chunk = &mut buf[..n * FRAME_BYTES];
            self.i2c.write_read(ADDR, &[REG_FIFO_R_W], chunk).await?;
            for f in chunk.chunks_exact(FRAME_BYTES) {
                sink(parse_frame(f));
            }
            frames -= n;
        }

        Ok(Drain::Drained)
    }

    async fn measure_gyro_bias(&mut self) -> Result<Option<[i16; 3]>, i2c::Error> {
        let mut skip = CAL_SKIP;
        let mut n: usize = 0;
        let mut sum = [0i32; 3];
        let mut min = [i16::MAX; 3];
        let mut max = [i16::MIN; 3];
        let mut polls = 0;

        self.reset_fifo().await?;
        while n < CAL_SAMPLES {
            polls += 1;
            if polls > CAL_MAX_POLLS {
                return Ok(None);
            }
            Timer::after_millis(20).await;
            self.drain(|s| {
                if skip > 0 {
                    skip -= 1;
                    return;
                }
                if n >= CAL_SAMPLES {
                    return;
                }
                for i in 0..3 {
                    let g = s.gyro[i];
                    sum[i] += g as i32;
                    min[i] = min[i].min(g);
                    max[i] = max[i].max(g);
                }
                n += 1;
            })
            .await?;
        }

        let limit = CAL_MAX_RANGE_DPS_X10 * GYRO_LSB_PER_DPS_X10 as i32 / 100;
        let still = (0..3).all(|i| (max[i] as i32 - min[i] as i32) <= limit);
        if !still {
            return Ok(None);
        }
        let mean = |i: usize| (sum[i] / n as i32) as i16;
        Ok(Some([mean(0), mean(1), mean(2)]))
    }

    pub async fn run(&mut self) -> ! {
        loop {
            if with_timeout(WAKE_TIMEOUT, self.int.wait_for_rising_edge())
                .await
                .is_err()
            {
                warn!(
                    "MPU6050 INT timeout: no data-ready edge for {} ms",
                    WAKE_TIMEOUT.as_millis()
                );
            }
            // newest FIFO sample is at most one period older than this instant
            let t_us = Instant::now().as_micros();

            let res = self
                .drain(|s| SHARED.lock(|c| c.borrow_mut().push(s, t_us)))
                .await;

            match res {
                Ok(Drain::Drained) => {}
                Ok(Drain::Overflow) => {
                    SHARED.lock(|c| {
                        let mut s = c.borrow_mut();
                        s.resets = s.resets.wrapping_add(1);
                        s.pending_gap = true;
                    });
                    warn!("MPU6050 FIFO overflow, FIFO reset");
                }
                Err(e) => {
                    SHARED.lock(|c| {
                        let mut s = c.borrow_mut();
                        s.errors = s.errors.wrapping_add(1);
                        s.pending_gap = true;
                    });
                    warn!("MPU6050 I2C error: {}", e);
                    Timer::after_millis(5).await;
                    if let Err(e) = self.reset_fifo().await {
                        warn!("MPU6050 FIFO reset failed: {}", e);
                        Timer::after_millis(50).await;
                    }
                }
            }
        }
    }
}

#[embassy_executor::task]
pub async fn mpu6050_task(mut mpu: Mpu6050) -> ! {
    mpu.run().await
}

/// Drop everything buffered so far. Called when the host first connects: samples taken while
/// nobody was listening are not a loss, so the first packet must not start with a gap.
pub fn discard_backlog() {
    SHARED.lock(|c| {
        let mut s = c.borrow_mut();
        s.head = 0;
        s.len = 0;
        s.pending_gap = false;
        s.lost = 0;
    });
}
