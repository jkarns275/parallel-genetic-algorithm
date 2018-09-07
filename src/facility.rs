use std::sync::{Arc, Mutex, RwLock, atomic::{AtomicBool, Ordering}, mpsc::Sender, Barrier};
use std::thread::JoinHandle;
use std::time::SystemTime;
use std::convert::TryInto;
use std::thread;
use std::cmp;

use rand::{prelude::Distribution, distributions::Standard, SeedableRng, Rng};

use constants::*;

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
    pub ty: T,
}

impl<T> Swap<T> where T:  Into<u64> + Clone + Copy {
    pub fn tx(&self) -> usize { self.tx.into() as usize }
    pub fn ty(&self) -> usize { self.ty.into() as usize }
    pub fn fx(&self) -> usize { self.fx.into() as usize }
    pub fn fy(&self) -> usize { self.fy.into() as usize }
}

#[derive(Clone)]
/// A chromosome is just an ordered series of swaps.
struct Chromosome<T> where T: Into<u64> + From<u64> + Clone + Copy {
    pub swaps: Vec<Swap<T>>
}

impl<T> Chromosome<T> where T: Into<u64> + From<u64> + Clone + Copy {

    pub fn new<R: Rng>(ncols: usize, nrows: usize, rng: &mut R) -> Self {
        let len = ncols * nrows;
        let mut ret = Chromosome { swaps: Vec::with_capacity(len) };
        for _ in 0..len {
            let tx = rng.gen::<u32>() as u64 % ncols as u64;
            let ty = rng.gen::<u32>() as u64 % nrows as u64;
            let fx = rng.gen::<u32>() as u64 % ncols as u64;
            let fy = rng.gen::<u32>() as u64 % nrows as u64;
            ret.swaps.push(Swap {   tx: tx.into(),
                                    ty: ty.into(),
                                    fx: fx.into(),
                                    fy: fy.into(), });
        }
        ret
    }

    /// Creates a new random swap and replaces one of the existing one. The replaced Swap and it's
    /// index are returned as a tuple.
    pub fn mutate<R: Rng>(&mut self, rng: &mut R, ncols: T, nrows: T) -> (Swap<T>, usize) {
        let tx = rng.gen::<u32>() as u64 % ncols.into();
        let ty = rng.gen::<u32>() as u64 % nrows.into();
        let fx = rng.gen::<u32>() as u64 % ncols.into();
        let fy = rng.gen::<u32>() as u64 % nrows.into();
        let index = rng.gen::<usize>() % self.swaps.len();
        let old_swap = self.swaps[index].clone();
        self.swaps[index] =
            Swap {  fx: fx.into(),
                    fy: fy.into(),
                    tx: tx.into(),
                    ty: ty.into(), };
        (old_swap, index)
    }
}

#[derive(Clone)]
pub struct Facility<T, R> where T: Affinity + Clone + Send + Sync, R: Rng {
    nrows: usize,
    ncols: usize,
    data: Arc<Vec<T>>,
    temp_data: Option<Vec<T>>,
    dna: Vec<Chromosome<u64>>,
    parents: Option<(Arc<RwLock<Facility<T, R>>>, Arc<RwLock<Facility<T, R>>>)>,
    rng: R,
    id: usize,
    original_fitness: Option<f64>
}

impl<T, R> Facility<T, R> where T: Affinity + Clone + Send + Sync, R: Rng {
    pub fn new(id: usize, ncols: usize, nrows: usize, data: Arc<Vec<T>>, rng: R) -> Self {
        Facility {
            dna: vec![],
            parents: None,
            original_fitness: None,
            temp_data: Some((*data).clone()),
            ncols,
            nrows,
            rng,
            data,
            id,
        }
    }

    pub fn set_parents(&mut self, parenta: Arc<RwLock<Self>>, parentb: Arc<RwLock<Self>>) {
        self.parents = Some((parenta, parentb));
    }

    /// Returns a tuple of (old fitness, new fitness).
    pub fn evolve(&mut self) {
        let chr_ind = self.rng.gen::<usize>() % self.dna.len();
        let (swap, ind) =
            self.dna[chr_ind].mutate(&mut self.rng, self.ncols as u64, self.nrows as u64);
    }

    pub fn gen_and_add_chromosome(&mut self) {
        self.dna.push(Chromosome::new(self.ncols, self.nrows, &mut self.rng));
    }

    pub fn replace_with_child(&mut self) {
        use std::borrow::Borrow;
        if let Some((parenta, parentb)) = self.parents.take() {
            let (ra, rb)  = (parenta.read(), parentb.read());
            match (ra, rb) {
                (Ok(pa), Ok(pb)) => {
                    //self.breed_middle_single_point(pa.borrow(), pb.borrow());
                    self.breed_rand_single_point(pa.borrow(), pb.borrow());
                },
                (Err(ea), _) => {
                    println!("Encountered error trying to lock parents ea: {:?}", ea);
                },
                (_, Err(eb)) => {
                    println!("Encountered error trying to lock parents ea: {:?}", eb);
                },
                (Err(ea), Err(eb)) => {
                    println!("Encountered errors trying to lock parents: {:?} {:?}", ea, eb);
                },
            }
        } else {
            println!("Failed to find parents to create child!");
        }
    }

    /// Replace this facility with the child of itself and other using single point crossover.
    pub fn breed_rand_single_point(&mut self, parenta: &Facility<T, R>, parentb: &Facility<T, R>) {
        debug_assert_eq!(parenta.dna.len(), self.dna.len());
        debug_assert_eq!(parentb.dna.len(), self.dna.len());

        let modulus = self.nrows * self.ncols;
        let nrows = self.nrows;
        for ((self_chr, a_chr), b_chr) in
            self.dna.iter_mut().zip(parenta.dna.iter()).zip(parentb.dna.iter()) {
            let mut index = nrows + (self.rng.gen::<usize>() % (modulus - nrows));
            self_chr.swaps[0..index].clone_from_slice(&a_chr.swaps[0..index]);
            self_chr.swaps[index..].clone_from_slice(&b_chr.swaps[index..]);
        }
    }

    pub fn breed_middle_single_point(&mut self, parenta: &Facility<T, R>, parentb: &Facility<T, R>) {
        debug_assert_eq!(parenta.dna.len(), self.dna.len());
        debug_assert_eq!(parentb.dna.len(), self.dna.len());

        let modulus = self.nrows * self.ncols;
        let nrows = self.nrows;
        for ((self_chr, a_chr), b_chr) in
            self.dna.iter_mut().zip(parenta.dna.iter()).zip(parentb.dna.iter()) {
            self_chr.swaps[0..modulus / 2].clone_from_slice(&a_chr.swaps[0..modulus / 2]);
            self_chr.swaps[modulus / 2..].clone_from_slice(&b_chr.swaps[modulus / 2..]);
        }
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
        // println!("col: {}, row: {}, index: {}", col, row, index);
        let mut fitness = 0.0;
        if row > 0 {
            let (r1, r2) = (&data[index],
                            &data[index - 1]);
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

    pub fn fitness(&mut self) -> (f64, f64, Vec<f64>) {
        let mut data = self.temp_data.take().unwrap();
        data.clone_from_slice(&self.data[..]);
        for chromosome in self.dna.iter() {
            for swap in chromosome.swaps.iter() {
                data.swap(swap.tx() * self.ncols + swap.ty(),
                          swap.fx() * self.ncols + swap.fy());
            }
        }

        // Swap in the "new" grid that has been changed according to the chromosomes
        let mut fitness_grid = vec![0.0; self.ncols * self.nrows];

        {
            let data = &mut data[..];

            for col in 1..self.ncols - 1 {
                let col_ind = col * self.nrows;
                for row in 1..self.nrows - 1 {
                    fitness_grid[col_ind + row] = self.fast_cell_fitness(data, col, row);
                }
            }

            // Taking into account the top and bottom row.
            let mut index = 0;
            let mut index2 = self.nrows - 1;
            for col in 0..self.ncols {
                fitness_grid[index] = self.slow_cell_fitness(data, col, 0);
                fitness_grid[index2] = self.slow_cell_fitness(data, col, self.nrows - 1);
                index += self.nrows;
                index2 += self.nrows;
            }

            // Taking into account the left and right columns (without the first and last cells of
            // the row, since they've been accounted for in the above code - we don't want them
            // to be counted twice).
            index = 1;
            index2 = (self.ncols - 1) * self.nrows + 1;
            for row in 1..self.nrows - 1 {
                fitness_grid[index] = self.slow_cell_fitness(data, 0, row);
                fitness_grid[index2] = self.slow_cell_fitness(data, self.ncols - 1, row);
                index += 1;
                index2 += 1;
            }
        }
        let _ = self.temp_data.get_or_insert(data);
        let (sum, grid) = (fitness_grid.iter().sum(), fitness_grid);
        match self.original_fitness.clone() {
            None => {
                let _ = self.original_fitness.get_or_insert(sum);
                (sum, sum, grid)
            },
            Some(x) => {
                (x, sum, grid)
            }
        }
    }
}

impl<T, R> Facility<T, R>
    where T: Affinity + Clone + Send + Sync,
          R: Rng,
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

use std::convert::TryFrom;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::fmt::Error;
use std::fmt::Display;

pub struct Fitness {
    pub sum: f64,
    pub grid: Vec<f64>,
    pub id: usize,
    pub original: f64,
}

impl Debug for Fitness {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "(id: {}, f: {:.2}, o: {:.2})", self.id, self.sum, self.original)
    }
}

impl Display for Fitness {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "(id: {}, sum: {})", self.id, self.sum)
    }
}

impl Eq for Fitness {}

impl PartialEq for Fitness {
    fn eq(&self, other: &Self) -> bool {
        self.sum == other.sum
    }

    fn ne(&self, other: &Self) -> bool {
        self.sum != other.sum
    }
}

impl PartialOrd for Fitness {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.sum.partial_cmp(&other.sum)
    }
}

impl Ord for Fitness {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        match self.partial_cmp(other) {
            Some(r) => r,
            None    => cmp::Ordering::Equal,
        }
    }
}

#[allow(unused)]
pub struct FacilityWorker<T, R>
    where T: 'static + Affinity + Clone + Send + Sync,
          R: 'static + Rng + SeedableRng + Send + Sync {
    /// A shared mutex for the facility. This will be read from and written during the breeding
    /// process (which is performed by the main thread).
    pub facility: Arc<RwLock<Facility<T, R>>>,

    /// A boolean flag that, when set to true, will eventually be read by the worker thread - the
    /// worker thread will then terminate.
    should_terminate: Arc<AtomicBool>,

    /// A shared container for the fitness to be sent through by the worker thread after an
    /// iteration.
    fitness_receiver: Arc<Mutex<Option<Fitness>>>,

    /// Just a thread handle. Will be used to ensure the thread gets terminated gracefully.
    thread_handle: Option<JoinHandle<()>>,

    /// A unique ID for this worker
    id: usize
}


impl<T, R> FacilityWorker<T, R>
    where T: 'static + Affinity + Clone + Send + Sync,
          R: 'static + Rng + SeedableRng + Send + Sync {

    pub fn new(id: usize, nchrs: usize, tick_barrier: Arc<Barrier>, done_sender: Sender<()>,
               ncols: usize, nrows: usize, data_base: Arc<Vec<T>>) -> Self {

        let now = SystemTime::UNIX_EPOCH;
        let seed128 = now.elapsed().unwrap().as_nanos() >> id as u128;
        let mut seed = <R as SeedableRng>::Seed::default();
        {
            println!("{}", seed128);
            use std::mem::transmute;
            let actual_seed_bytes = seed.as_mut() ;
            let len = actual_seed_bytes.len();
            if len > 16 {
                actual_seed_bytes[0..16]
                    .clone_from_slice(unsafe { &transmute::<u128, [u8; 16]>(seed128)[..] });
            } else {
                actual_seed_bytes[..]
                    .clone_from_slice(unsafe { &transmute::<u128, [u8; 16]>(seed128)[..len] });
            }
        }
        let facility = Arc::new(RwLock::new(
            Facility::new(id, ncols, nrows, data_base, R::from_seed(seed))));
        let should_terminate = Arc::new(AtomicBool::new(false));
        let fitness_receiver = Arc::new(Mutex::new(None));

        let thread_handle = Some({
            let should_terminate = should_terminate.clone();
            let fitness_receiver = fitness_receiver.clone();
            let facility = facility.clone();

            thread::spawn(move || {
                if let Ok(mut f) = facility.try_write() {
                    for _ in 0..nchrs {
                        f.gen_and_add_chromosome();
                    }
                } else {
                    println!("Failed to obtain facility mutex in thread {}", id);
                }


                // Unwrapping is okay-ish here, since the only time this will fail is when the
                // main thread has panicked
                done_sender.send(()).unwrap(); // Count D
                tick_barrier.wait(); // Sync A
                loop {
                    if (*should_terminate).load(Ordering::Relaxed) { return }
                    if let Ok(mut f) = facility.try_write() {
                        for i in 0..N_MUTATIONS { f.evolve(); }
                        let (original, sum, grid) = f.fitness();
                        if let Ok(mut fitness_recv) = fitness_receiver.try_lock() {
                            assert!(fitness_recv.take().is_none());
                            let _ =
                                fitness_recv
                                .get_or_insert(Fitness { sum, original, grid, id, });
                        } else {
                            println!("Failed to obtain fitness receiver mutex in thread {}", id);
                        }
                    } else {
                        println!("Failed to obtain facility mutex in thread {}", id);
                    }

                    for i in 0..N_GENS_PER_ITER {
                        // Unwrapping here is okay-ish for the same reason as above.
                        done_sender.send(()).unwrap(); // Count C

                        // Sync E
                        tick_barrier.wait();

                        // Do breeding...
                        // Breeding is done in the main thread as of right now...
                        let has_parents;
                        if let Ok(f) = facility.read() {
                            has_parents = f.parents.is_some();
                        } else {
                            has_parents = false;
                            println!("Failed to obtain facility mutex for breeding in thread {}", id);
                        }

                        if has_parents {
                            if let Ok(mut f) = facility.write() {
                                f.replace_with_child();
                            } else {
                                println!("Failed to obtain facility mutex for breeding in thread {}", id);
                            }
                        }

                        // Sync F
                        tick_barrier.wait();

                        if i == N_GENS_PER_ITER - 1 { break }
                        if let Ok(mut f) = facility.write() {
                            let (original, sum, grid) = f.fitness();
                            if let Ok(mut fitness_recv) = fitness_receiver.lock() {
                                assert!(fitness_recv.take().is_none());
                                let _ =
                                    fitness_recv
                                    .get_or_insert(Fitness { sum, grid, original, id, });

                            }
                        }
                    }

                    tick_barrier.wait(); // Sync B
                }
            })
        });

        FacilityWorker {
            facility,
            should_terminate,
            thread_handle,
            fitness_receiver,
            id
        }
    }

    pub fn get_fitness(&mut self) -> Fitness {
        if let Ok(mut t) = self.fitness_receiver.lock() {
            t.take().unwrap()
        } else {
            panic!("Failed to lock fitness_receiver in thread {}", self.id)
        }
    }

    pub fn set_to_terminate(&mut self) {
        self.should_terminate.store(true, Ordering::Relaxed);
    }

    pub fn join(mut self) -> Result<(), ()> {
        let thread_handle = self.thread_handle.take().unwrap();
        match thread_handle.join() {
            Ok(())  => Ok(()),
            Err(_)  => Err(())
        }
    }
}