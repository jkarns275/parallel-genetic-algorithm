use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}, mpsc::{Receiver, Sender}, Barrier};
use std::thread::JoinHandle;
use rand::{SeedableRng, Rng};
use core::mem;
use std::thread;
use std::time::SystemTime;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use std::convert::TryInto;

pub trait Affinity {
    fn aff_impl(&self, other: &Self) -> f64;
    fn affinity(&self, other: &Self) -> f64 { f64::abs(self.aff_impl(other)) }
}

impl Affinity for f64 {
    fn aff_impl(&self, other: &Self) -> f64 { self - other }
}

pub trait FromRng {
    fn from_rng<R: Rng>(rng: &mut R) -> Self;
}

#[derive(Clone)]
/// A generic representation of a swap - the cell at (fx, fy) should be swapped with (tx, ty).
struct Swap<T> where T: Into<u64> + Clone + Copy {
    pub fx: T,
    pub fy: T,
    pub tx: T,
    pub ty: T
}

impl<T> Swap<T> where T: Into<u64> + Clone + Copy {
    fn fx(&self) -> usize { self.fx.into() as usize }
    fn fy(&self) -> usize { self.fy.into() as usize }
    fn tx(&self) -> usize { self.tx.into() as usize }
    fn ty(&self) -> usize { self.ty.into() as usize }
}

#[derive(Clone)]
/// A chromosome is just an ordered series of swaps.
struct Chromosome<T> where T: Into<u64> + TryFrom<u64> + Clone + Copy {
    pub swaps: Vec<Swap<T>>
}

impl<T> Chromosome<T> where T: Into<u64> + TryFrom<u64> + Clone + Copy {
    pub fn new(capacity: usize) -> Self {
        Chromosome { swaps: Vec::with_capacity(capacity) }
    }

    /// Creates a new random swap and replaces one of the existing one. The replaced Swap and it's
    /// index are returned as a tuple.
    pub fn mutate<R: Rng>(&mut self, rng: &mut R, ncols: T, nrows: T) -> (Swap<T>, usize) {
        let (tx, ty) = (rng.next_u32() as u64 % ncols.into(), rng.next_u32() as u64 % nrows.into());
        let (fx, fy) = (rng.next_u32() as u64 % ncols.into(), rng.next_u32() as u64 % nrows.into());
        let index = rng.gen::<usize>() % self.swaps.len();
        let old_swap = self.swaps[index].clone();
        self.swaps[index] =
            Swap {  tx: tx.try_into().unwrap_or(panic!("")),
                    ty: ty.try_into().unwrap_or(panic!("")),
                    fx: fx.try_into().unwrap_or(panic!("")),
                    fy: fy.try_into().unwrap_or(panic!("")) };
        (old_swap, index)
    }

    pub fn restore_mutation(&mut self, swap: Swap<T>, index: usize) { self.swaps[index] = swap; }
}

#[derive(Clone)]
pub struct Facility<T, R> where T: Affinity + Clone + Send + Sync, R: Rng + Send + Sync {
    nrows: usize,
    ncols: usize,
    data: Arc<Vec<T>>,
    temp_data: Option<Vec<T>>,
    dna: Vec<Chromosome<u8>>,
    rng: R
}

impl<T, R> Facility<T, R> where T: Affinity + Clone + Send + Sync, R: Rng + Send + Sync {
    pub fn new(ncols: usize, nrows: usize, data: Arc<Vec<T>>, rng: R) -> Self {
        Facility {
            dna: vec![],
            temp_data: Some((*data).clone()),
            ncols,
            nrows,
            rng,
            data,
        }
    }

    /// Returns a tuple of (old fitness, new fitness).
    pub fn evolve(&mut self) -> (f64, f64) {
        let old_fitness = self.fitness();
        let mut new_fitness;
        loop {
            let chr_ind = self.rng.gen::<usize>() % self.dna.len();
            let (swap, ind) = self.dna[chr_ind].mutate(&mut self.rng, self.ncols as u8, self.nrows as u8);
            new_fitness = self.fitness();
            if new_fitness > old_fitness { break; }
            self.dna[chr_ind].restore_mutation(swap, ind);
        }

        (old_fitness, new_fitness)
    }

    /// Replace this facility with the child of itself and other using single point crossover.
    pub fn breed_single_point(&mut self, other: &Facility<T, R>) {
        debug_assert_eq!(other.dna.len(), self.dna.len());
        let modulus = (self.nrows * self.ncols);
        for (self_chr, other_chr) in self.dna.iter_mut().zip(other.dna.iter()) {
            let index = self.rng.gen::<usize>() % modulus;
            self_chr.swaps[index..].clone_from_slice(&other_chr.swaps[index..]);
        }
    }

    fn cell_at<'a>(&'a self, col: usize, row: usize) -> &'a T {
        &self.data[col * self.nrows + row]
    }

    fn fast_cell_fitness(&self, data: &[T], col: usize, row: usize) -> f64 {
        debug_assert_ne!(col, 0);
        debug_assert_ne!(col, self.ncols - 1);
        debug_assert_ne!(row, 0);
        debug_assert_ne!(row, self.nrows - 1);
        let nrows = self.nrows;
        let index = nrows * col + row;
        let refs = [&data[index], &data[index + nrows], &data[index - nrows], &data[index + 1], &self.data[index - 1]];
        let cell: &T = refs[0];
        refs[1..].iter().map(|other_ref| cell.affinity(other_ref)).sum()
    }

    /// Fitness is calculated by summing the affinity of the four directly-adjacent cells. Since
    /// grid-bordering cells have only 2 or 3 neighbors, special care must be given to them when
    /// calculating fitness. This function is more or less just bounds checking.
    fn slow_cell_fitness(&self, data: &[T], col: usize, row: usize) -> f64 {
        let nrows = self.nrows;
        let ncols = self.ncols;
        let index = col * nrows + row;
        let mut fitness = 0.0;
        if row > 0 {
            let (r1, r2) = (&data[index], &data[index - 1]);
            fitness += r1.affinity(r2);
        }
        if row < nrows - 1 {
            let (r1, r2) = (&data[index], &data[index + 1]);
            fitness += r1.affinity(r2);
        }
        if col > 0 {
            let (r1, r2) = (&data[index], &data[index - nrows]);
            fitness += r1.affinity(r2);
        }
        if col < ncols - 1 {
            let (r1, r2) = (&data[index], &data[index + nrows]);
            fitness += r1.affinity(r2);
        }
        fitness
    }

    pub fn fitness(&mut self) -> f64 {
        let mut data = self.temp_data.take().unwrap();
        for chromosome in self.dna.iter() {
            for swap in chromosome.swaps.iter() {
                data.swap(swap.tx() * self.ncols + swap.ty(),
                          swap.fx() * self.ncols + swap.fy());
            }
        }

        // Swap in the "new" grid that has been changed according to the chromosomes
        let mut fitness = 0.0;

        {
            let mut data = &mut data[..];

            for col in 1..self.ncols - 1 {
                for row in 1..self.nrows - 1 {
                    fitness += self.fast_cell_fitness(data, col, row);
                }
            }

            for col in 0..self.ncols {
                fitness += self.slow_cell_fitness(data, col, 0);
                fitness += self.slow_cell_fitness(data, col, self.nrows - 1);
            }

            for row in 1..self.nrows - 1 {
                fitness += self.slow_cell_fitness(data, 0, row);
                fitness += self.slow_cell_fitness(data, self.ncols, row);
            }
        }
        let _ = self.temp_data.get_or_insert(data);

        fitness
    }
}

impl<T, R> Facility<T, R>
    where T: Affinity + Clone + Send + Sync,
          R: Rng + Send + Sync,
          Standard: Distribution<T> {
    pub fn gen_random_data(ncols: usize, nrows: usize, rng: R) -> Vec<T> {
        let mut rng = rng;
        let mut data = Vec::with_capacity(nrows * ncols);
        for _ in 0..nrows * ncols {
            data.push(Standard.sample(&mut rng));
        }
        data
    }
}

use std::marker::PhantomData;
use std::convert::TryFrom;

pub struct FacilityWorker<T, R>
    where T: 'static + Affinity + Clone + Send + Sync,
          R: 'static + Rng + SeedableRng + Send + Sync {
    /// A shared mutex for the facility. This will be read from and written during the breeding
    /// process (which is performed by the main thread).
    facility: Arc<Mutex<Facility<T, R>>>,

    /// A boolean flag that, when set to true, will eventually be read by the worker thread - the
    /// worker thread will then terminate.
    should_terminate: Arc<AtomicBool>,

    /// A shared container for the fitness to be sent through by the worker thread after an
    /// iteration.
    fitness_receiver: Arc<Mutex<Option<f64>>>,

    /// Just a thread handle. Will be used to ensure the thread gets terminated gracefully.
    thread_handle: JoinHandle<()>,

    /// A unique ID for this FacilityWorker
    id: usize,
}

const N_MUTATIONS: u32 = 16;

impl<T, R> FacilityWorker<T, R>
    where T: 'static + Affinity + Clone + Send + Sync,
          R: 'static + Rng + SeedableRng + Send + Sync {
    pub fn new(id: usize, tick_barrier: Arc<Barrier>, done_sender: Sender<()>, ncols: usize,
               nrows: usize, data_base: Arc<Vec<T>>) -> Self {

        let now = SystemTime::UNIX_EPOCH;
        let seed128 = now.elapsed().unwrap().as_nanos();
        let mut seed = <R as SeedableRng>::Seed::default();
        {
            use std::mem::transmute;
            let actual_seed_bytes = seed.as_mut();
            let len = actual_seed_bytes.len();
            if len > 16 {
                actual_seed_bytes[0..16]
                    .clone_from_slice(unsafe { &transmute::<u128, [u8; 16]>(seed128)[..] });
            } else {
                actual_seed_bytes[..]
                    .clone_from_slice(unsafe { &transmute::<u128, [u8; 16]>(seed128)[..len] });
            }
        }
        let facility = Arc::new(Mutex::new(Facility::new(ncols, nrows, data_base, R::from_seed(seed))));
        let should_tick = Arc::new(AtomicBool::new(false));
        let should_terminate = Arc::new(AtomicBool::new(false));
        let fitness_receiver = Arc::new(Mutex::new(None));

        let thread_handle = {
            let should_terminate = should_terminate.clone();
            let fitness_receiver = fitness_receiver.clone();
            let facility = facility.clone();

            thread::spawn(move || {
                done_sender.send(()); // Count D
                tick_barrier.wait(); // Sync A
                loop {

                    if (*should_terminate).load(Ordering::Relaxed) { return }
                    if let Ok(mut f) = (*facility).lock() {
                        for i in 0..N_MUTATIONS { f.evolve(); }
                    }
                    done_sender.send(()); // Count C
                    tick_barrier.wait(); // Sync B
                }
            })
        };

        FacilityWorker {
            facility,
            should_terminate,
            thread_handle,
            fitness_receiver,
            id,
        }
    }
}