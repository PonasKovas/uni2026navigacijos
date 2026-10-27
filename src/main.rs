#![no_std]
#![no_main]

use crate::{
    mpu6050::{
        ACCEL_LSB_PER_G, GYRO_LSB_PER_DPS_X10, MAX_BATCH, Mpu6050, SAMPLE_RATE_HZ, Sample,
        discard_backlog, mpu6050_task, take_batch,
    },
    vl53l5cx::Vl53l5cx,
};
use defmt::{error, info, unwrap, warn};
use embassy_executor::Spawner;
use embassy_stm32::{
    Config, bind_interrupts,
    exti::ExtiInput,
    gpio::{Level, Output, OutputType, Pull, Speed},
    i2c::{self, I2c},
    interrupt::typelevel::{EXTI9_5, EXTI15_10},
    peripherals::{DMA1_CH2, DMA1_CH3, DMA1_CH6, DMA1_CH7, I2C3, TIM2, USART2},
    rcc::{Pll, PllMul, PllPreDiv, PllRDiv, PllSource, Sysclk},
    time::{Hertz, khz},
    timer::{
        low_level::CountingMode,
        simple_pwm::{PwmPin, SimplePwm},
    },
    usart::Uart,
};
use embassy_time::{Duration, Timer};
use zerocopy::{Immutable, IntoBytes, KnownLayout};

use {defmt_rtt as _, panic_probe as _};

mod mpu6050;
mod vl53l5cx;

bind_interrupts!(struct Irqs {
    USART2 => embassy_stm32::usart::InterruptHandler<USART2>;
    DMA1_CHANNEL7 => embassy_stm32::dma::InterruptHandler<DMA1_CH7>;
    DMA1_CHANNEL6 => embassy_stm32::dma::InterruptHandler<DMA1_CH6>;

    EXTI15_10 => embassy_stm32::exti::InterruptHandler<EXTI15_10>;

    EXTI9_5 => embassy_stm32::exti::InterruptHandler<EXTI9_5>;
    I2C3_EV => embassy_stm32::i2c::EventInterruptHandler<I2C3>;
    I2C3_ER => embassy_stm32::i2c::ErrorInterruptHandler<I2C3>;
    DMA1_CHANNEL2 => embassy_stm32::dma::InterruptHandler<DMA1_CH2>;
    DMA1_CHANNEL3 => embassy_stm32::dma::InterruptHandler<DMA1_CH3>;
});

// ------------------------------------------------------------------ wire format
//
// One packet per request byte (0x23). All little-endian.
//
//   Header (56 B) | zones: 64 x 4 x i16 (512 B) | n_imu x Sample (14 B: ax ay az gx gy gz temp, i16) | crc32 (4 B)
//
// crc32 = zlib/IEEE CRC-32 over everything before it (Python: zlib.crc32(pkt[:-4])).
// Python header format: "<4sIQQIIHHHHHHH3hHH"
//
// imu_t_us is the estimated MCU time of the LAST imu sample; sample i is at
// imu_t_us - (n_imu - 1 - i) / imu_rate_hz. The MPU's clock is not the MCU's clock, so the
// host should fit the true rate over a long window from (imu_seq0 + n_imu, imu_t_us) pairs.
// vl53_t_us is the MCU time of the VL53 data-ready edge (imu_t_us is normally <= one IMU
// period before it).
// temp is the MPU die temperature: °C = temp / 340 + 36.53.

const REQ_BYTE: u8 = 0x23;

const FLAG_IMU_GAP_BEFORE: u16 = 1 << 0; // data lost right before the first IMU sample: don't integrate across it
const FLAG_IMU_CLIPPED: u16 = 1 << 1; // an axis hit full scale in this batch
const FLAG_IMU_BACKLOG: u16 = 1 << 2; // more samples are waiting, next packet continues
const FLAG_IMU_BIAS_VALID: u16 = 1 << 3; // gyro_bias was measured while still

#[derive(IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct Header {
    magic: [u8; 4], // b"NAV2" (NAV1 had 12 B samples without temp)
    vl53_seq: u32,
    vl53_t_us: u64,
    imu_t_us: u64,
    imu_seq0: u32, // absolute index of the first IMU sample; must equal previous seq0 + n unless FLAG_IMU_GAP_BEFORE
    imu_lost: u32, // cumulative: dropped because the MCU ring was full
    n_imu: u16,
    flags: u16,
    imu_rate_hz: u16,
    imu_resets: u16, // cumulative MPU FIFO overflow resets
    imu_errors: u16, // cumulative MPU I2C errors
    accel_lsb_per_g: u16,
    gyro_lsb_per_dps_x10: u16, // LSB/dps * 10
    gyro_bias: [i16; 3],       // raw LSB, measured at boot (subtract from gyro)
    imu_misaligned: u16, // cumulative MPU FIFO resets after a frame failed the alignment check
    reserved: u16,
}

const HEADER_LEN: usize = size_of::<Header>();
const ZONES_LEN: usize = size_of::<[[i16; 4]; 64]>();
const CRC_LEN: usize = 4;
const MAX_PACKET_LEN: usize = HEADER_LEN + ZONES_LEN + MAX_BATCH * size_of::<Sample>() + CRC_LEN;
const _: () = assert!(HEADER_LEN == 56);

const fn make_crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}
static CRC_TABLE: [u32; 256] = make_crc_table();

fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = CRC_TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    !c
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let mut config = Config::default();
    // setup clock to be 80MHz (using 16MHz HSI and 5x PLL)
    config.rcc.hsi = true;
    config.rcc.msi = None;
    config.rcc.sys = Sysclk::PLL1_R; // 80MHz
    config.rcc.pll = Some(Pll {
        source: PllSource::HSI,
        prediv: PllPreDiv::DIV1,
        mul: PllMul::MUL10,
        divp: None,
        divq: None,
        divr: Some(PllRDiv::DIV2),
    });
    config.rcc.hsi48 = None;

    let p = embassy_stm32::init(config);

    drop((p.PA5, p.PA6)); // must not be used because of SB16 and SB18 solder bridges

    //////////////////////
    // PERIPHERALS SETUP

    info!("Hello");

    let led_pwm = SimplePwm::new(
        p.TIM2,
        None,
        Some(PwmPin::new(p.PB3, OutputType::PushPull)),
        None,
        None,
        Hertz(1000),
        CountingMode::EdgeAlignedUp,
    );

    unwrap!(spawner.spawn(led_task(led_pwm)));

    let mut vl53l5cx = {
        let int = ExtiInput::new(
            p.PA12, // D2
            p.EXTI12,
            Pull::Up,
            Irqs,
        );
        // everything starts low: the sensor stays unpowered until init() runs the power-up sequence
        let i2c_rst = Output::new(
            p.PB0, // D3
            Level::Low,
            Speed::Low,
        );
        let lpn = Output::new(
            p.PB1, // D6
            Level::Low,
            Speed::Low,
        );
        let pwr_en = Output::new(
            p.PC14, // D7
            Level::Low,
            Speed::Low,
        );
        let mut i2c_conf = i2c::Config::default();
        i2c_conf.frequency = khz(400);
        i2c_conf.gpio_speed = Speed::High;
        i2c_conf.sda_pullup = false;
        i2c_conf.scl_pullup = false;
        i2c_conf.timeout = Duration::from_millis(10_000);
        let i2c = I2c::new_blocking(
            p.I2C1, p.PB6, // D5
            p.PB7, // D4
            i2c_conf,
        );

        Vl53l5cx::init(int, i2c_rst, lpn, pwr_en, i2c).await
    };

    let imu = {
        let int = ExtiInput::new(
            p.PB5, // D11, MPU INT: push-pull, active high
            p.EXTI5,
            Pull::Down, // a missing wire reads low instead of floating
            Irqs,
        );

        let mut i2c_conf = i2c::Config::default();
        i2c_conf.frequency = khz(400);
        i2c_conf.gpio_speed = Speed::High;
        i2c_conf.sda_pullup = false;
        i2c_conf.scl_pullup = false;
        i2c_conf.timeout = Duration::from_millis(50);
        let i2c = I2c::new(
            p.I2C3, p.PA7, // A6
            p.PB4, // D12
            p.DMA1_CH2, p.DMA1_CH3, Irqs, i2c_conf,
        );

        unwrap!(Mpu6050::init(i2c, int).await)
    };
    unwrap!(spawner.spawn(mpu6050_task(imu)));

    let mut uart_config = embassy_stm32::usart::Config::default();
    uart_config.baudrate = 921_600;
    let mut uart = unwrap!(Uart::new(
        p.USART2,
        p.PA15,
        p.PA2,
        p.DMA1_CH7,
        p.DMA1_CH6,
        Irqs,
        uart_config,
    ));

    let mut zones = [[0i16; 4]; 64];
    let mut samples = [Sample::ZERO; MAX_BATCH];
    let mut packet = [0u8; MAX_PACKET_LEN];
    let mut vl53_seq: u32 = 0;
    let mut first_request = true;
    let (mut prev_lost, mut prev_resets, mut prev_misaligned, mut prev_errors) = (0u32, 0u16, 0u16, 0u16);

    info!("ready");

    loop {
        // wait for uart ready signal
        let mut buf = [0u8; 1];
        loop {
            match uart.read(&mut buf).await {
                Ok(()) if buf[0] == REQ_BYTE => break,
                Ok(()) => warn!("UART unknown signal received: 0x{:x}", buf[0]),
                Err(e) => warn!("UART rx error: {}", e),
            }
        }

        // samples taken while nobody was listening are not a loss
        if first_request {
            discard_backlog();
            first_request = false;
        }

        let (result, vl53_t) = vl53l5cx.next_frame().await;

        for z in 0..64 {
            for t in 0..4 {
                let val = if [5, 9].contains(&unwrap!(result.target_status(z, t))) {
                    unwrap!(result.distance_mm(z, t))
                } else {
                    -1
                };
                zones[z][t] = val;
            }
        }

        // The VL53 readout blocks the executor right after its data-ready edge, so the ring
        // holds IMU data up to ~that edge. Anything newer goes out in the next packet.
        let batch = take_batch(&mut samples);
        let n = batch.n;

        let mut flags = 0u16;
        if batch.gap_before {
            flags |= FLAG_IMU_GAP_BEFORE;
            warn!(
                "IMU gap before seq {}: dropped +{}, fifo resets +{}, misaligned +{}, i2c errors +{}",
                batch.seq0,
                batch.lost.wrapping_sub(prev_lost),
                batch.resets.wrapping_sub(prev_resets),
                batch.misaligned.wrapping_sub(prev_misaligned),
                batch.errors.wrapping_sub(prev_errors),
            );
        }
        (prev_lost, prev_resets, prev_misaligned, prev_errors) =
            (batch.lost, batch.resets, batch.misaligned, batch.errors);
        if batch.clipped {
            flags |= FLAG_IMU_CLIPPED;
        }
        if batch.backlog {
            flags |= FLAG_IMU_BACKLOG;
        }
        if batch.calib_ok {
            flags |= FLAG_IMU_BIAS_VALID;
        }

        let header = Header {
            magic: *b"NAV2",
            vl53_seq,
            vl53_t_us: vl53_t.as_micros(),
            imu_t_us: batch.t_us,
            imu_seq0: batch.seq0,
            imu_lost: batch.lost,
            n_imu: n as u16,
            flags,
            imu_rate_hz: SAMPLE_RATE_HZ,
            imu_resets: batch.resets,
            imu_errors: batch.errors,
            accel_lsb_per_g: ACCEL_LSB_PER_G,
            gyro_lsb_per_dps_x10: GYRO_LSB_PER_DPS_X10,
            gyro_bias: batch.gyro_bias,
            imu_misaligned: batch.misaligned,
            reserved: 0,
        };

        let mut off = 0;
        packet[off..off + HEADER_LEN].copy_from_slice(header.as_bytes());
        off += HEADER_LEN;
        packet[off..off + ZONES_LEN].copy_from_slice(zones.as_bytes());
        off += ZONES_LEN;
        let imu_bytes = samples[..n].as_bytes();
        packet[off..off + imu_bytes.len()].copy_from_slice(imu_bytes);
        off += imu_bytes.len();
        let crc = crc32(&packet[..off]);
        packet[off..off + CRC_LEN].copy_from_slice(&crc.to_le_bytes());
        off += CRC_LEN;

        if let Err(e) = uart.write(&packet[..off]).await {
            error!("Error writing to uart: {}", e);
        }

        vl53_seq = vl53_seq.wrapping_add(1);
    }
}

#[embassy_executor::task]
async fn led_task(mut led_pwm: SimplePwm<'static, TIM2>) {
    led_pwm.ch2().enable();

    let max = led_pwm.ch2().max_duty_cycle() as f32;
    let mut brightness = 0.0f32;
    let mut step = 0.005f32;

    loop {
        brightness += step;
        if brightness >= 1.0 {
            brightness = 1.0;
            step = -step;
        } else if brightness <= 0.0 {
            brightness = 0.0;
            step = -step;
        }

        // `brightness * brightness` gives smooth sine-like breathing
        // and matches human eye perception better than linear fading.
        led_pwm
            .ch2()
            .set_duty_cycle((brightness * brightness * max) as u32);
        Timer::after_millis(4).await;
    }
}
