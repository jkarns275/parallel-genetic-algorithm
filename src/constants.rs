pub const N_THREADS: usize = 8;
pub const N_THREADS_2: usize = N_THREADS / 2;
pub const N_GENERATIONS: usize = !0;
pub const N_COLS: usize = 48;
pub const N_ROWS: usize = 48;
pub const N_EMPTY_CELLS: usize = 40 * 48;
pub const N_CELLS: usize = (N_COLS * N_ROWS) - N_EMPTY_CELLS;

pub const N_CHRS: usize = 128;
pub const N_PARENTS: usize = 2;
pub const MUTATION_COEFFICIENT: f64 = 0.00125;
pub const N_GENS_PER_ITER: usize = 8;
pub const UPDATE_FREQUENCY: usize = 164;
