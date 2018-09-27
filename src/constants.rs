pub const N_THREADS: usize = 16;
pub const N_THREADS_2: usize = N_THREADS / 2;
pub const N_GENERATIONS: usize = 1024 * 1024 * 1024;
pub const N_COLS: usize = 8;
pub const N_ROWS: usize = 8;
pub const N_EMPTY_CELLS: usize = 58;
pub const N_CELLS: usize = (N_COLS * N_ROWS) - N_EMPTY_CELLS;

pub const N_CHRS: usize = 124;
pub const N_PARENTS: usize = 3;
pub const MUTATION_COEFFICIENT: f64 = 0.01;
pub const N_GENS_PER_ITER: usize = 8;
