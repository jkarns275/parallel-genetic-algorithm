#![feature(duration_as_u128)]
#![feature(int_to_from_bytes)]
#![feature(try_from)]

extern crate rand;
extern crate core;

pub mod facility;
use facility::*;

use std::thread;
use std::sync::mpsc::channel;
use std::sync::{Arc, Barrier};
use rand::ThreadRng;
use rand::XorShiftRng;

fn main() {
    const N_THREADS: usize = 8;
    const N_GENERATIONS: usize = 1024 * 16;
    const N_COLS: usize = 32;
    const N_ROWS: usize = 32;
    const N_CHRS: usize = 8;

    use rand::thread_rng;
    use std::time::SystemTime;
    use std::mem::transmute;
    use rand::SeedableRng;

    let seed = SystemTime::UNIX_EPOCH.elapsed().unwrap().as_nanos();
    let rng = XorShiftRng::from_seed(unsafe { transmute::<u128, [u8; 16]>(seed) });
    let data = Arc::new(Facility::<f64, XorShiftRng>::gen_random_data(NCOLS, NROWS, rng));

    let (tx, rx) = channel::<()>();

    let barrier = Arc::new(Barrier::new(N_THREADS + 1));
    let workers = (0..N_THREADS).map(|id| {
        FacilityWorker::<f64, XorShiftRng>::new(id, N_CHRS, barrier.clone(), tx.clone(), N_COLS,
                                                N_ROWS, data.clone())
    }).collect::<Vec<_>>();

    let mut counter = 0;
    // Count D
    while counter < N_THREADS {
        match rx.recv() {
            Ok(()) => counter += 1,
            Err(_) => panic!("Received error in channel."),
        }
    }

    barrier.wait(); // Sync A

    for i in 0..N_GENERATIONS {
        let mut counter = 0;

        // Count C
        while counter < N_THREADS {
            match rx.recv() {
                Ok(()) => counter += 1,
                Err(_) => panic!("Received error in channel.")
            }
        }

        // Do breeding

        barrier.wait(); // Sync B
    }

}
