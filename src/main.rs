#![feature(duration_as_u128)]
#![feature(int_to_from_bytes)]
#![feature(try_from)]


extern crate rand;
extern crate core;
extern crate pancurses;
use pancurses::*;

pub mod facility;
pub mod constants;

use constants::*;
use facility::*;

use std::sync::mpsc::channel;
use std::sync::{Arc, Barrier};
use rand::{Rng, XorShiftRng};

fn main() {
    type Affine = (f64, f64, f64);

    const CHARSET: [char; 6] = ['-', '~', '#', '$', '!', '█'];
    const COLORS: [i16; 6] = [0, 1, 2, 3, 4, 5];

    let affine_max = <Affine as Affinity>::max() * 4.0;

    let window = initscr();
    noecho();
    window.keypad(false);
    window.nodelay(true);
    start_color();

    init_pair(0, COLOR_BLACK, COLOR_BLACK );
    init_pair(1, COLOR_RED, COLOR_BLACK);
    init_pair(2, COLOR_YELLOW, COLOR_BLACK);
    init_pair(3, COLOR_MAGENTA, COLOR_BLACK);
    init_pair(4, COLOR_CYAN, COLOR_BLACK);
    init_pair(5, COLOR_GREEN, COLOR_BLACK);

    use std::time::SystemTime;
    use std::mem::transmute;
    use rand::SeedableRng;

    let seed = SystemTime::UNIX_EPOCH.elapsed().unwrap().as_nanos();
    let rng = XorShiftRng::from_seed(unsafe { transmute::<u128, [u8; 16]>(seed) });
    let data = Arc::new(Facility::<Affine, XorShiftRng>::gen_random_data(N_COLS, N_ROWS, rng));
    let (tx, rx) = channel::<()>();

    let barrier = Arc::new(Barrier::new(N_THREADS + 1));
    let mut workers = (0..N_THREADS).map(|id| {
        FacilityWorker::<Affine, XorShiftRng>::new(id, barrier.clone(), tx.clone(), N_COLS,
                                                N_ROWS, data.clone())
    }).collect::<Vec<_>>();

    let mut rng = XorShiftRng::from_seed(unsafe { transmute::<u128, [u8; 16]>(seed) });

    let mut counter = 0;
    let mut end = false;
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
        let mut weights = workers.iter_mut().map(|w| w.get_fitness()).collect::<Vec<_>>();
        // Unstable sort is faster according to rustdocs
        weights.sort_unstable();

        // println!("Fitnesses: {:?}", weights);

        // The lowest (first) N_THREADS / 2 will be thrown out. The other half will be selected for
        // breeding randomly. Every facility of the other half will have at least one child.
        let _ = (0..N_THREADS - N_PARENTS)
            .map(|i| (i, N_THREADS - N_PARENTS + (rng.gen::<usize>() % N_PARENTS),
                      N_THREADS - N_PARENTS + (rng.gen::<usize>() % N_PARENTS)))
            .map(|(i, a, b)| {
                if a == b { (i, a, N_THREADS - N_PARENTS + ((b + 1) % N_PARENTS)) } else { (i, a, b) }
            })
            .map(|(i, a, b)| {
                let (ind, inda, indb) = (weights[i].id, weights[a].id, weights[b].id);
                let parenta = workers[inda].facility.clone();
                let parentb = workers[indb].facility.clone();
                if let Ok(mut c) = workers[ind].facility.try_write() {
                    c.set_parents(parenta, parentb);
                } else {
                    println!("Failed to obtain parenta lock of thread {}", inda);
                }
            })
            .count();
        let max_fitness = weights.pop().unwrap();
        // Sync E - after this call to wait the actual breeding will occur in the worker threads
        barrier.wait();

        fn index(col: usize, row: usize) -> usize {
            col * N_ROWS + row
        }

        let aff_to_chclr = |aff: f64| -> (char, i16) {
            let aff = ((aff / affine_max) * 10.0) as usize;
            if aff >= 6 {
                (CHARSET[5], COLORS[5])
            } else {
                (CHARSET[aff], COLORS[aff])
            }
        };

        for row in 0..N_ROWS {
            window.mv(row as i32, 0);
            for col in 0..N_COLS {
                let (_ch, clr) = aff_to_chclr(max_fitness.grid[index(col, row)]);
                window.color_set(clr);
                window.addch('█');
            }
        }
        window.color_set(2);
        let df = max_fitness.sum - max_fitness.original;
        if df > 0.0 {
            window.printw(format!("\nΔ fitness: +{:.3}", max_fitness.sum - max_fitness.original));
        } else {
            window.printw(format!("\nΔ fitness: {:.3}", max_fitness.sum - max_fitness.original));
        }
        window.printw(format!("\n actual fitness: {:.3}", max_fitness.sum));
        match window.getch() {
            Some(Input::Character(x)) if x == ' ' => end = true,
            _ => {}
        }
        window.refresh();
        // Update display.
        // TODO: actually have a display. Possible just display the "most fit" grid
        if end { break }
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

    let _ = workers
        .into_iter()
        .map(|w| w.join().unwrap())
        .count();

    endwin();
}
