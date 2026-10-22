use defmt::{info, panic, unwrap};
use embassy_stm32::{
    exti::ExtiInput,
    gpio::Output,
    i2c::{I2c, Master},
    mode,
};
use embassy_time::{Delay, Instant, Timer};
use vl53l5cx_rs::{PowerMode, Ranging, RangingMode, Resolution, Results};

pub struct Vl53l5cx {
    int: ExtiInput<'static, mode::Async>,
    _i2c_rst: Output<'static>,
    _lpn: Output<'static>,
    _pwr_en: Output<'static>,
    uld: vl53l5cx_rs::Vl53l5cx<I2c<'static, mode::Blocking, Master>, Delay, Ranging>,
}

impl Vl53l5cx {
    pub async fn init(
        int: ExtiInput<'static, mode::Async>,
        mut i2c_rst: Output<'static>,
        mut lpn: Output<'static>,
        mut pwr_en: Output<'static>,
        i2c: I2c<'static, mode::Blocking, Master>,
    ) -> Self {
        // Full power cycle, same order as ST's VL53L5CX_Reset_Sensor. An MCU reset/reflash does
        // not reset the sensor, so it can still be mid-transaction and holding SDA low.
        lpn.set_low();
        i2c_rst.set_low();
        pwr_en.set_low();
        Timer::after_millis(250).await; // rails fully discharged, I2C state gone

        pwr_en.set_high();
        Timer::after_millis(20).await;
        lpn.set_high();
        Timer::after_millis(100).await; // boot time

        let uld = vl53l5cx_rs::Vl53l5cx::new(i2c, Delay);

        info!("Initializing Vl53l5cx...");

        let mut uld = match uld.init() {
            Ok(x) => x,
            Err((_, e)) => panic!("Vl53l5cx initialization failed: {}", e),
        };

        unwrap!(uld.set_resolution(Resolution::R8x8));
        unwrap!(uld.set_ranging_frequency_hz(15));
        unwrap!(uld.set_ranging_mode(RangingMode::Continuous));
        unwrap!(uld.set_power_mode(PowerMode::Wakeup));

        let uld = unwrap!(uld.start_ranging().map_err(|e| e.1));

        Self {
            int,
            _i2c_rst: i2c_rst,
            _lpn: lpn,
            _pwr_en: pwr_en,
            uld,
        }
    }
    pub async fn next_frame(&mut self) -> (Results<'_>, Instant) {
        self.int.wait_for_falling_edge().await;
        let t = Instant::now();

        let data = unwrap!(self.uld.get_ranging_data());

        (data, t)
    }
}
