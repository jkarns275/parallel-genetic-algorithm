mod chromosome;
mod affinity;
mod worker;
mod weight;

pub use facility::weight::*;
pub use facility::worker::*;
pub use facility::affinity::*;

use std::sync::Arc;
use std::sync::RwLock;
use rand::Rng;
use facility::chromosome::Chromosome;
use constants::MUTATION_COEFFICIENT;
use core::cmp;

#[derive(Clone)]
pub struct Facility<T, R, W>
    where T: Affinity + Clone + Send + Sync,
          R: Rng,
          W: WeightGen {
    nrows: usize,
    ncols: usize,
    data: Arc<Vec<T>>,
    temp_data: Option<Vec<T>>,
    dna: Vec<Chromosome<u64>>,
    pub parents: Option<(Arc<RwLock<Facility<T, R, W>>>, Arc<RwLock<Facility<T, R, W>>>)>,
    rng: R,
    id: usize,
    original_fitness: Option<f64>,
    weight_gen: W,
}

impl<T, R, W> Facility<T, R, W>
    where T: Affinity + Clone + Send + Sync,
          R: Rng,
          W: WeightGen {
    pub fn new(id: usize, ncols: usize, nrows: usize, data: Arc<Vec<T>>, rng: R, weight_gen: W) -> Self {
        Facility {
            temp_data: Some((*data).clone()),
            original_fitness: None,
            parents: None,
            dna: vec![],
            weight_gen,
            ncols,
            nrows,
            data,
            rng,
            id,
        }
    }

    pub fn set_parents(&mut self, parenta: Arc<RwLock<Self>>, parentb: Arc<RwLock<Self>>) {
        self.parents = Some((parenta, parentb));
    }

    /// Returns a tuple of (old fitness, new fitness).
    pub fn evolve(&mut self) {
        let chr_ind = self.rng.gen::<usize>() % self.dna.len();
        let (_swap, _ind) =
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
                (Err(ea), Ok(_)) => {
                    println!("Encountered error trying to lock parents ea: {:?}", ea);
                },
                (Ok(_), Err(eb)) => {
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
    pub fn breed_rand_single_point(&mut self, parenta: &Facility<T, R, W>, parentb: &Facility<T, R, W>) {
        debug_assert_eq!(parenta.dna.len(), self.dna.len());
        debug_assert_eq!(parentb.dna.len(), self.dna.len());

        let modulus = self.nrows * self.ncols;
        let nrows = self.nrows;
        for ((self_chr, a_chr), b_chr) in
            self.dna.iter_mut().zip(parenta.dna.iter()).zip(parentb.dna.iter()) {
            let mut index = nrows + (self.rng.gen::<usize>() % (modulus - nrows));
            self_chr.swaps[0..index].clone_from_slice(&a_chr.swaps[0..index]);
            self_chr.swaps[index..].clone_from_slice(&b_chr.swaps[index..]);
            for i in cmp::max(index as isize - nrows as isize, 0) as usize .. cmp::min(index + nrows, modulus) {
                if self.rng.gen::<f64>() < MUTATION_COEFFICIENT {
                    self_chr.mutate_at(i, &mut self.rng, self.ncols as u64, self.nrows as u64);
                }
            }
        }
    }

    pub fn breed_middle_single_point(&mut self, parenta: &Facility<T, R, W>, parentb: &Facility<T, R, W>) {
        debug_assert_eq!(parenta.dna.len(), self.dna.len());
        debug_assert_eq!(parentb.dna.len(), self.dna.len());

        let modulus = self.nrows * self.ncols;
        for ((self_chr, a_chr), b_chr) in
            self.dna.iter_mut().zip(parenta.dna.iter()).zip(parentb.dna.iter()) {
            self_chr.swaps[0..modulus / 2].clone_from_slice(&a_chr.swaps[0..modulus / 2]);
            self_chr.swaps[modulus / 2..].clone_from_slice(&b_chr.swaps[modulus / 2..]);
        }
    }

    fn fast_cell_fitness(&self, data: &[T], col: usize, row: usize) -> f64 {
        if data[col * self.nrows + row].is_empty() { return 0.0 }

        fn manhattan_distance(c1: usize, r1: usize, c2: usize, r2: usize) -> f64 {
            ((c1 as isize - c2 as isize) as f64).abs() + ((r1 as isize - r2 as isize) as f64).abs()
        }

        let weight = self.weight_gen.weight(col, row);
        if weight > 0.0 {
            let cell = &data[col * self.nrows + row];
            let mut sum = 0.0;
            for r in 0..self.nrows {
                for c in 0..self.ncols {
                    if c == col && r == row { continue }
                    let md = manhattan_distance(col, row, c, r);
                    sum += cell.affinity(&data[c * self.nrows + r]) / (md * md);
                }
            }
            sum * weight
        } else {
            0.0
        }
    }

    // Calculates the fitness and returns
    // (original fitness, current fitness, layout, per-cell fitness)
    pub fn fitness(&mut self) -> (f64, f64, Vec<T>, Vec<f64>) {
        let mut data = self.temp_data.take().unwrap();
        data.clone_from_slice(&self.data[..]);
        for chromosome in self.dna.iter() {
            for swap in chromosome.swaps.iter() {
                data.swap(swap.tx() * self.nrows + swap.ty(),
                          swap.fx() * self.nrows + swap.fy());
            }
        }

        // Swap in the "new" grid that has been changed according to the chromosomes
        let mut fitness_grid = vec![0.0; self.ncols * self.nrows];

        {
            let data = &mut data[..];

            for col in 0..self.ncols {
                let col_ind = col * self.nrows;
                for row in 0..self.nrows {
                    fitness_grid[col_ind + row] = self.fast_cell_fitness(data, col, row);
                }
            }
        }

        let layout = data.clone();
        let _ = self.temp_data.get_or_insert(data);
        let (sum, grid) = (fitness_grid.iter().sum(), fitness_grid);
        match self.original_fitness.clone() {
            None => {
                let _ = self.original_fitness.get_or_insert(sum);
                (sum, sum, layout, grid)
            },
            Some(x) => {
                (x, sum, layout, grid)
            }
        }
    }

    pub fn gen_random_data(ncols: usize, nrows: usize, rng: R) -> Vec<T> {
        let mut rng = rng;
        let mut data = Vec::with_capacity(nrows * ncols);
        for _ in 0..nrows * ncols {
            data.push(T::gen(&mut rng));
        }
        data
    }
}

