#![no_std]
#![no_main]

use defmt::unwrap;
use embassy_executor::Spawner;
use embassy_futures::select::{Either, select};
use embassy_stm32::{
    Config, bind_interrupts,
    flash::Flash,
    gpio::{Flex, Level, Output, OutputType, Speed},
    peripherals::{self, DMA1_CH6, DMA1_CH7, USART2},
    rcc::{Pll, PllMul, PllPreDiv, PllRDiv, PllSource, Sysclk, mux::Adcsel},
    time::Hertz,
    timer::{
        low_level::CountingMode,
        simple_pwm::{PwmPin, SimplePwm},
    },
    usart::Uart,
};
use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, signal::Signal};
use embassy_time::Timer;

use {defmt_rtt as _, panic_probe as _};

bind_interrupts!(struct Irqs {
    USART2 => embassy_stm32::usart::InterruptHandler<USART2>;
    DMA1_CHANNEL7 => embassy_stm32::dma::InterruptHandler<DMA1_CH7>;
    DMA1_CHANNEL6 => embassy_stm32::dma::InterruptHandler<DMA1_CH6>;
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

    //////////////////////
    // PERIPHERALS SETUP

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

    let uart_config = embassy_stm32::usart::Config::default();
    let uart = unwrap!(Uart::new(
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
