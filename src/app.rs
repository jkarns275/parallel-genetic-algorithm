use constants::*;
use facility::*;

use std::sync::mpsc::channel;
use std::sync::{Arc, Barrier};
use rand::{Rng, XorShiftRng};
use std::time::Instant;

use std::time::SystemTime;
use std::mem::transmute;
use rand::SeedableRng;
use std::sync::mpsc::Receiver;
use pancurses::{self, *};

const CHARSET: [char; 6] = ['-', '~', '#', '$', '!', '█'];
const COLORS: [i16; 6] = [0, 1, 2, 3, 4, 5];

pub struct App<T, R> where T: 'static + Affinity + Clone + Send + Sync,
                       R: 'static + Rng + SeedableRng + Send + Sync {
    rng:        XorShiftRng,
    workers:    Vec<FacilityWorker<T, R>>,
    barrier:    Arc<Barrier>,
    count_recv: Receiver<()>,
    window:     Window,
    ms_per_iter: f64
}

impl<T, R> App<T, R> where  T: Affinity + Clone + Send + Sync,
                            R: Rng + SeedableRng + Send + Sync {

    const MS_PER_ITER_UPDATE_FREQ: usize = 256;

    pub fn new() -> Self {
        let window = App::<T, R>::init_window();
        let seed = SystemTime::UNIX_EPOCH.elapsed().unwrap().as_nanos();
        let rng = XorShiftRng::from_seed(unsafe { transmute::<u128, [u8; 16]>(seed) });
        let data = Arc::new(Facility::<T, XorShiftRng>::gen_random_data(N_COLS, N_ROWS, rng));
        let (count_send, count_recv) = channel::<()>();
        let barrier = Arc::new(Barrier::new(N_THREADS + 1));
        let workers = (0..N_THREADS).map(|id| {
            FacilityWorker::<T, R>::new( id, barrier.clone(), count_send.clone(), N_COLS,
                                         N_ROWS, data.clone())
        }).collect::<Vec<_>>();
        let rng = XorShiftRng::from_seed(unsafe { transmute::<u128, [u8; 16]>(seed) });
        let ms_per_iter: f64 = 0.0;

        App {
            count_recv,
            barrier,
            workers,
            window,
            rng,
            ms_per_iter
        }
    }

    fn init_window() -> Window {
        let window = initscr();

        noecho();

        window.keypad(false);
        window.nodelay(true);

        start_color();

        init_pair(0, COLOR_BLACK,   COLOR_BLACK );
        init_pair(1, COLOR_RED,     COLOR_BLACK);
        init_pair(2, COLOR_YELLOW,  COLOR_BLACK);
        init_pair(3, COLOR_MAGENTA, COLOR_BLACK);
        init_pair(4, COLOR_CYAN,    COLOR_BLACK);
        init_pair(5, COLOR_GREEN,   COLOR_BLACK);

        window
    }

    pub fn update_display(&mut self, max_fitness: Fitness, affine_max: f64) {
        fn indexof(col: usize, row: usize) -> usize { col * N_ROWS + row }

        let aff_to_chclr = |aff: f64| -> (char, i16) {
            let aff = (aff / affine_max) as usize;
            if aff >= 6 {
                (CHARSET[5], COLORS[5])
            } else {
                (CHARSET[aff], COLORS[aff])
            }
        };

        for row in 0..N_ROWS {
            self.window.mv(row as i32, 0);
            for col in 0..N_COLS {
                let (_ch, clr) = aff_to_chclr(max_fitness.grid[indexof(col, row)]);
                self.window.color_set(clr);
                self.window.addch('█');
            }
        }
        self.window.color_set(2);
        let df = max_fitness.sum - max_fitness.original;
        if df > 0.0 {
            self.window.printw(
                format!("\n  Δ fitness:       +{:.3}", max_fitness.sum - max_fitness.original));
        } else {
            self.window.printw(
                format!("\n  Δ fitness:       {:.3}", max_fitness.sum - max_fitness.original));
        }
        self.window.printw(
                format!("\n  actual fitness:  {:.3}", max_fitness.sum));
        self.window.printw(
                format!("\n  ms / iter:       {:.3}", self.ms_per_iter));
    }

    pub fn retrieve_fitnesses(&mut self) -> Vec<Fitness> {
        let mut weights = self.workers.iter_mut().map(|w| w.get_fitness()).collect::<Vec<_>>();
        weights.sort_unstable();
        weights
    }

    pub fn select_parents(&mut self, fitnesses: &mut Vec<Fitness>) {
        for i in 0..N_THREADS - N_PARENTS {
            let (i, a, b) = (i, N_THREADS - N_PARENTS + (self.rng.gen::<usize>() % N_PARENTS),
                            N_THREADS - N_PARENTS + (self.rng.gen::<usize>() % N_PARENTS));
            let (i, a, b) =
                if a == b {
                    (i, a, N_THREADS - N_PARENTS + ((b + 1) % N_PARENTS))
                } else {
                    (i, a, b)
                };

            let (ind, inda, indb) = (fitnesses[i].id, fitnesses[a].id, fitnesses[b].id);
            let parenta = self.workers[inda].facility.clone();
            let parentb = self.workers[indb].facility.clone();
            if let Ok(mut c) = self.workers[ind].facility.try_write() {
                c.set_parents(parenta, parentb);
            } else {
                println!("Failed to obtain child lock of thread {}", ind);
            }
        }
    }

    pub fn set_workers_to_terminate(&mut self) {
        for worker in self.workers.iter_mut() {
            worker.set_to_terminate();
        }
    }

    pub fn join_worker_threads(&mut self) {
        for worker in self.workers.drain(..).into_iter() {
            worker.join().unwrap();
        }
    }

    pub fn counter_wait(&mut self) {
        let mut counter = 0;
        // Count D
        while counter < N_THREADS {
            match self.count_recv.recv() {
                Ok(()) => counter += 1,
                Err(_) => panic!("Received error in channel."),
            }
        }
    }

    pub fn should_exit(&mut self) -> bool {
        match self.window.getch() {
            Some(Input::Character(x)) if x == ' ' => true,
            _ => false
        }
    }

    pub fn run(mut self) {
        let affine_max = <T as Affinity>::max() * 2 as f64;

        self.counter_wait();
        self.barrier.wait(); // Sync A

        let mut end = false;
        let mut start = Instant::now();

        for i in 0..N_GENERATIONS {
            self.counter_wait();

            let mut fitnesses = self.retrieve_fitnesses();
            self.select_parents(&mut fitnesses);

            // Sync E - after this call to wait the actual breeding will occur in the worker threads
            self.barrier.wait();

            if i % Self::MS_PER_ITER_UPDATE_FREQ == 0 {
                self.ms_per_iter =
                    start.elapsed().as_nanos() as f64 / 1.0e6 / (Self::MS_PER_ITER_UPDATE_FREQ as f64);
                start = Instant::now();
            }

            let max_fitness = fitnesses.pop().unwrap();
            self.update_display(max_fitness, affine_max);

            if self.should_exit() {
                end = true
            }

            // Update display.
            self.window.refresh();

            if end { break }

            // If we're on the last iteration, we want to set all the the threads to terminate before we
            // synchronize with them and they continue on the the next iteration.
            if i < N_GENERATIONS - 1 {
                self.barrier.wait(); // Sync B
            }
        }

        self.set_workers_to_terminate();

        // Sync B
        self.barrier.wait();

        self.join_worker_threads();

        self.window.delwin();
        pancurses::endwin();
    }
}
