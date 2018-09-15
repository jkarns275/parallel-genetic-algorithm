use facility::affinity::Affinity;
use std::sync::Arc;
use std::sync::RwLock;
use rand::Rng;
use facility::chromosome::Chromosome;
use constants::MUTATION_COEFFICIENT;
use core::cmp;

#[derive(Clone)]
pub struct Facility<T, R> where T: Affinity + Clone + Send + Sync, R: Rng {
    nrows: usize,
    ncols: usize,
    data: Arc<Vec<T>>,
    temp_data: Option<Vec<T>>,
    dna: Vec<Chromosome<u64>>,
    pub parents: Option<(Arc<RwLock<Facility<T, R>>>, Arc<RwLock<Facility<T, R>>>)>,
    rng: R,
    id: usize,
    original_fitness: Option<f64>,
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
            for i in cmp::max(index as isize - nrows as isize, 0) as usize .. cmp::min(index + nrows, modulus) {
                if self.rng.gen::<f64>() < MUTATION_COEFFICIENT {
                    self_chr.mutate_at(i, &mut self.rng, self.ncols as u64, self.nrows as u64);
                }
            }
        }
    }

    pub fn breed_middle_single_point(&mut self, parenta: &Facility<T, R>, parentb: &Facility<T, R>) {
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
                data.swap(swap.tx() * self.nrows + swap.ty(),
                          swap.fx() * self.nrows + swap.fy());
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

    pub fn gen_random_data(ncols: usize, nrows: usize, rng: R) -> Vec<T> {
        let mut rng = rng;
        let mut data = Vec::with_capacity(nrows * ncols);
        for _ in 0..nrows * ncols {
            data.push(T::gen(&mut rng));
        }
        data
    }
}
