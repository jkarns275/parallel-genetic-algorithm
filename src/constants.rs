pub const N_THREADS: usize = 8;
pub const N_THREADS_2: usize = N_THREADS / 2;
pub const N_GENERATIONS: usize = !0;
pub const N_COLS: usize = 128;
pub const N_ROWS: usize = 64;
pub const N_EMPTY_CELLS: usize = 60 * 128;
pub const N_CELLS: usize = (N_COLS * N_ROWS) - N_EMPTY_CELLS;

pub const N_CHRS: usize = 128;
pub const N_PARENTS: usize = 3;
pub const MUTATION_COEFFICIENT: f64 = 0.0025;
pub const N_GENS_PER_ITER: usize = 1;
pub const UPDATE_FREQUENCY: usize = 1;
