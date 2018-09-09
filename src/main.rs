#![feature(duration_as_u128)]
#![feature(int_to_from_bytes)]
#![feature(try_from)]

extern crate rand;
extern crate core;

pub mod facility;
pub mod constants;

use constants::*;
use facility::*;

use std::sync::mpsc::channel;
use std::sync::{Arc, Barrier};
use rand::{Rng, XorShiftRng};

fn main() {


    use std::time::SystemTime;
    use std::mem::transmute;
    use rand::SeedableRng;

    let seed = SystemTime::UNIX_EPOCH.elapsed().unwrap().as_nanos();
    let rng = XorShiftRng::from_seed(unsafe { transmute::<u128, [u8; 16]>(seed) });
    let data = Arc::new(Facility::<f64, XorShiftRng>::gen_random_data(N_COLS, N_ROWS, rng));
    let (tx, rx) = channel::<()>();

    let barrier = Arc::new(Barrier::new(N_THREADS + 1));
    let mut workers = (0..N_THREADS).map(|id| {
        FacilityWorker::<f64, XorShiftRng>::new(id, N_CHRS, barrier.clone(), tx.clone(), N_COLS,
                                                N_ROWS, data.clone())
    }).collect::<Vec<_>>();

    let mut rng = XorShiftRng::from_seed(unsafe { transmute::<u128, [u8; 16]>(seed) });

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
        for iter in 0..N_GENS_PER_ITER {
            let mut counter = 0;
            // Count C
            while counter < N_THREADS {
                match rx.recv() {
                    Ok(()) => counter += 1,
                    Err(_) => panic!("Received error in channel.")
                }
            }

            // Do breeding
            let mut weights = workers.iter_mut().map(|w| w.get_fitness()).collect::<Vec<_>>();
            // Unstable sort is faster according to rustdocs
            weights.sort_unstable();
            if iter == N_GENS_PER_ITER - 1 {
                println!("Fitnesses: {:?}", weights);
            }
            // The lowest (first) N_THREADS / 2 will be thrown out. The other half will be selected for
            // breeding randomly. Every facility of the other half will have at least one child.
            let partners = (0..N_THREADS - N_PARENTS)
                .map(|i| (i, N_THREADS - N_PARENTS + (rng.gen::<usize>() % N_PARENTS),
                          N_THREADS - N_PARENTS + (rng.gen::<usize>() % N_PARENTS)))
                .map(|(i, a, b)| {
                    if a == b { (i, a, N_THREADS - N_PARENTS + ((b + 1) % N_PARENTS)) } else { (i, a, b) }
                })
                .map(|(i, a, b)| {
                    let (ind, inda, indb) = (weights[i].id, weights[a].id, weights[b].id);
                    let parenta = workers[inda].facility.clone();
                    let parentb = workers[indb].facility.clone();
                    let child = workers[ind].facility.clone();
                    if let Ok(mut c) = workers[ind].facility.try_write() {
                        c.set_parents(parenta, parentb);
                    } else {
                        println!("Failed to obtain parenta lock of thread {}", inda);
                    }
                })
                .count();

            // Sync E - after this call to wait the actual breeding will occur in the worker threads
            barrier.wait();
            // Sync F
            barrier.wait();
        }

        // Update display.
        // TODO: actually have a display. Possible just display the "most fit" grid

        // If we're on the last iteration, we want to set all the the threads to terminate before we
        // synchronize with them and they continue on the the next iteration.
        if i < N_GENERATIONS - 1 {
            barrier.wait(); // Sync B
        }
    }

    for worker in workers.iter_mut() {
        worker.set_to_terminate();
    }

    // Sync B
    barrier.wait();

    let facilities = workers
        .into_iter()
        .map(|w| w.join().unwrap())
        .collect::<Vec<_>>();
}
