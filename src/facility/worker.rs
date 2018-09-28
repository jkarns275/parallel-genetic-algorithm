use std::fmt::Debug;
use std::fmt::Formatter;
use std::fmt::Error;
use std::fmt::Display;
use std::cmp;
use facility::affinity::Affinity;
use rand::Rng;
use rand::SeedableRng;
use facility::Facility;
use std::sync::RwLock;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::SystemTime;
use std::sync::mpsc::Sender;
use std::sync::Barrier;
use std::thread;
use std::sync::atomic::Ordering;
use constants::*;
use facility::weight::WeightGen;

pub struct Fitness<T> {
    pub sum: f64,
    pub grid: Vec<f64>,
    pub id: usize,
    pub original: f64,
    pub layout: Vec<T>
}

impl<T> Debug for Fitness<T> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "(id: {}, f: {:.2}, o: {:.2})", self.id, self.sum, self.original)
    }
}

impl<T> Display for Fitness<T> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "(id: {}, sum: {})", self.id, self.sum)
    }
}

impl<T> Eq for Fitness<T> {}

impl<T> PartialEq for Fitness<T> {
    fn eq(&self, other: &Self) -> bool {
        self.sum == other.sum
    }

    fn ne(&self, other: &Self) -> bool {
        self.sum != other.sum
    }
}

impl<T> PartialOrd for Fitness<T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.sum.partial_cmp(&other.sum)
    }
}

impl<T> Ord for Fitness<T> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        match self.partial_cmp(other) {
            Some(r) => r,
            None    => cmp::Ordering::Equal,
        }
    }
}

#[allow(unused)]
pub struct FacilityWorker<T, R, W>
    where T: 'static + Affinity + Clone + Send + Sync,
          R: 'static + Rng + SeedableRng + Send + Sync,
          W: 'static + WeightGen + Send + Sync {
    /// A shared mutex for the facility. This will be read from and written during the breeding
    /// process (which is performed by the main thread).
    pub facility: Arc<RwLock<Facility<T, R, W>>>,

    /// A boolean flag that, when set to true, will eventually be read by the worker thread - the
    /// worker thread will then terminate.
    should_terminate: Arc<AtomicBool>,

    /// A shared container for the fitness to be sent through by the worker thread after an
    /// iteration.
    fitness_receiver: Arc<Mutex<Option<Fitness<T>>>>,

    /// Just a thread handle. Will be used to ensure the thread gets terminated gracefully.
    thread_handle: Option<JoinHandle<()>>,

    /// A unique ID for this worker
    id: usize
}

struct Worker<T, R, W>
    where T: 'static + Affinity + Clone + Send + Sync,
          R: 'static + Rng + SeedableRng + Send + Sync,
          W: 'static + WeightGen + Send + Sync {
    fitness_receiver:   Arc<Mutex<Option<Fitness<T>>>>,
    facility:           Arc<RwLock<Facility<T, R, W>>>,
    should_terminate:   Arc<AtomicBool>,
    tick_barrier:       Arc<Barrier>,
    done_sender:        Sender<()>,
    id: usize,
}

impl<T, R, W> Worker<T, R, W>
    where T: 'static + Affinity + Clone + Send + Sync,
          R: 'static + Rng + SeedableRng + Send + Sync,
          W: 'static + WeightGen + Send + Sync {

    pub fn calculate_and_send_fitness(&mut self) {
        if let Ok(mut f) = self.facility.try_write() {
            let (original, sum, layout, grid) = f.fitness();
            if let Ok(mut fitness_recv) = self.fitness_receiver.try_lock() {
                assert!(fitness_recv.take().is_none());
                let _ = fitness_recv
                        .get_or_insert(Fitness { sum, original, grid, layout, id: self.id, });
            } else {
                println!("Failed to obtain fitness receiver mutex in thread {}", self.id);
            }
        } else {
            println!("Failed to obtain facility mutex in thread {}", self.id);
        }
    }

    pub fn should_terminate(&self) -> bool { (*self.should_terminate).load(Ordering::Relaxed) }

    pub fn do_breeding(&mut self) {
        let has_parents;
        if let Ok(f) = self.facility.read() {
            has_parents = f.parents.is_some();
        } else {
            has_parents = false;
            println!("Failed to obtain facility mutex for breeding in thread {}", self.id);
        }

        if has_parents {
            if let Ok(mut f) = self.facility.write() {
                f.replace_with_child();
            } else {
                println!("Failed to obtain facility mutex for breeding in thread {}", self.id);
            }
        }
    }

    pub fn run(mut self) {
        if let Ok(mut f) = self.facility.try_write() {
            for _ in 0..N_CHRS {
                f.gen_and_add_chromosome();
            }
        } else {
            println!("Failed to obtain facility mutex in thread {}", self.id);
        }

        // Unwrapping is okay-ish here, since the only time this will fail is when the
        // main thread has panicked
        self.done_sender.send(()).unwrap(); // Count D
        self.tick_barrier.wait(); // Sync A

        loop {
            if self.should_terminate() { return }

            self.calculate_and_send_fitness();

            // Unwrapping here is okay-ish for the same reason as above.
            self.done_sender.send(()).unwrap(); // Count C
            // Sync E
            self.tick_barrier.wait();

            // Parent selection is done in the main thread as of right now...
            self.do_breeding();

            self.tick_barrier.wait(); // Sync B
        }
    }
}


impl<T, R, W> FacilityWorker<T, R, W>
    where T: 'static + Affinity + Clone + Send + Sync,
          R: 'static + Rng + SeedableRng + Send + Sync,
          W: 'static  + WeightGen + Send + Sync {

    pub fn new(id: usize, tick_barrier: Arc<Barrier>, done_sender: Sender<()>,
               ncols: usize, nrows: usize, data_base: Arc<Vec<T>>, weight_gen: W) -> Self {

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
            Facility::new(id, ncols, nrows, data_base, R::from_seed(seed), weight_gen)));
        let should_terminate = Arc::new(AtomicBool::new(false));
        let fitness_receiver = Arc::new(Mutex::new(None));

        let thread_handle = Some({
            let should_terminate = should_terminate.clone();
            let fitness_receiver = fitness_receiver.clone();
            let facility = facility.clone();

            thread::spawn(move || {
                Worker {
                    facility,
                    should_terminate,
                    fitness_receiver,
                    id,
                    tick_barrier,
                    done_sender,
                }.run()
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

    pub fn get_fitness(&mut self) -> Fitness<T> {
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
