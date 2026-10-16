#![no_std]
#![no_main]

use defmt::{info, unwrap};
use embassy_executor::Spawner;
use embassy_stm32::{
    Config, bind_interrupts,
    exti::ExtiInput,
    gpio::{Level, Output, OutputType, Pull, Speed},
    i2c::{self, I2c},
    interrupt::typelevel::EXTI15_10,
    peripherals::{DMA1_CH6, DMA1_CH7, USART2},
    rcc::{Pll, PllMul, PllPreDiv, PllRDiv, PllSource, Sysclk},
    time::{Hertz, khz},
    timer::{
        low_level::CountingMode,
        simple_pwm::{PwmPin, SimplePwm},
    },
    usart::Uart,
};
use embassy_time::{Duration, Timer};

use crate::vl53l5cx::Vl53l5cx;

use {defmt_rtt as _, panic_probe as _};

mod vl53l5cx;

bind_interrupts!(struct Irqs {
    USART2 => embassy_stm32::usart::InterruptHandler<USART2>;
    DMA1_CHANNEL7 => embassy_stm32::dma::InterruptHandler<DMA1_CH7>;
    DMA1_CHANNEL6 => embassy_stm32::dma::InterruptHandler<DMA1_CH6>;
    EXTI15_10 => embassy_stm32::exti::InterruptHandler<EXTI15_10>;
});

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

    let mut led_pwm = SimplePwm::new(
        p.TIM2,
        None,
        Some(PwmPin::new(p.PB3, OutputType::PushPull)),
        None,
        None,
        Hertz(1000),
        CountingMode::EdgeAlignedUp,
    );

    led_pwm.ch2().enable();

    {
        let int = ExtiInput::new(
            p.PA12, // D2
            p.EXTI12,
            Pull::Up,
            Irqs,
        );
        let i2c_rst = Output::new(
            p.PB0, // D3
            Level::Low,
            Speed::Low,
        );
        let lpn = Output::new(
            p.PB1, // D6
            Level::High,
            Speed::Low,
        );
        let pwr_en = Output::new(
            p.PC14, // D7
            Level::High,
            Speed::Low,
        );
        let mut i2c_conf = i2c::Config::default();
        i2c_conf.frequency = khz(100);
        i2c_conf.gpio_speed = Speed::High;
        i2c_conf.sda_pullup = false;
        i2c_conf.scl_pullup = false;
        i2c_conf.timeout = Duration::from_millis(10_000);
        let i2c = I2c::new_blocking(
            p.I2C1, p.PB6, // D5
            p.PB7, // D4
            i2c_conf,
        );

        let vl53l5cx = Vl53l5cx::init(int, i2c_rst, lpn, pwr_en, i2c).await;

        unwrap!(spawner.spawn(test(vl53l5cx)));
    }

    let uart_config = embassy_stm32::usart::Config::default();
    let _uart = unwrap!(Uart::new(
        p.USART2,
        p.PA15,
        p.PA2,
        p.DMA1_CH7,
        p.DMA1_CH6,
        Irqs,
        uart_config,
    ));

    let mut x = 0.0;
    loop {
        x += 0.01;
        let duty_cycle = (libm::sin(x) * 0.5 + 0.5) * led_pwm.ch2().max_duty_cycle() as f64;
        led_pwm.ch2().set_duty_cycle(duty_cycle as u32);
        Timer::after_millis(1).await;
    }
}

#[embassy_executor::task]
async fn test(mut vl53l5cx: Vl53l5cx) {
    loop {
        vl53l5cx.next_frame().await;
    }
}
