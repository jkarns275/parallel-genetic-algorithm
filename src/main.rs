#![feature(duration_as_u128)]
#![feature(int_to_from_bytes)]
#![feature(try_from)]


extern crate rand;
extern crate core;
extern crate pancurses;

pub mod facility;
pub mod constants;
pub mod app;

use app::App;

fn main() {
    type Affine = (f64, f64);
    type Rng = XorShiftRng;

    App::<Affine, Rng>::new().run();
}
