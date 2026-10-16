use defmt::{info, panic, unwrap};
use embassy_stm32::{
    exti::ExtiInput,
    gpio::Output,
    i2c::{I2c, Master},
    mode,
};
use embassy_time::{Delay, Timer};
use vl53l5cx_rs::{PowerMode, Ranging, RangingMode, Resolution};

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
        i2c_rst.set_low();
        lpn.set_high();
        pwr_en.set_high();

        Timer::after_millis(10).await;

        let uld = vl53l5cx_rs::Vl53l5cx::new(i2c, Delay);

        info!("Initializing Vl53l5cx...");

        let mut uld = match uld.init() {
            Ok(x) => x,
            Err((_, e)) => panic!("Vl53l5cx initialization failed: {}", e),
        };

        unwrap!(uld.set_resolution(Resolution::R8x8));
        unwrap!(uld.set_ranging_frequency_hz(1));
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
    pub async fn next_frame(&mut self) {
        self.int.wait_for_falling_edge().await;

        let data = unwrap!(self.uld.get_ranging_data());

        let mut sum: i64 = 0;
        let mut nones = 0;
        for i in 0..64 {
            sum += data.distance_mm(i, 0).unwrap_or(5000) as i64;
            if data.distance_mm(i, 0).is_none() {
                nones += 1;
            }
        }

        info!(
            "zones: {}\nsilicon_temp_degc: {}\ndist_sum_mm: {}\nnones: {}",
            data.zones(),
            data.silicon_temp_degc(),
            sum,
            nones
        );
    }
}
