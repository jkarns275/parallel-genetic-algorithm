use rand::Rng;

#[derive(Clone)]
/// A generic representation of a swap - the cell at (fx, fy) should be swapped with (tx, ty).
pub struct Swap<T> where T: Into<u64> + Clone + Copy {
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
pub struct Chromosome<T> where T: Into<u64> + From<u64> + Clone + Copy {
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

    pub fn mutate_at<R: Rng>(&mut self, index: usize, rng: &mut R, ncols: T, nrows: T) -> (Swap<T>, usize) {
        let tx = rng.gen::<u32>() as u64 % ncols.into();
        let ty = rng.gen::<u32>() as u64 % nrows.into();
        let fx = rng.gen::<u32>() as u64 % ncols.into();
        let fy = rng.gen::<u32>() as u64 % nrows.into();
        let old_swap = self.swaps[index].clone();
        self.swaps[index] =
            Swap {  fx: fx.into(),
                    fy: fy.into(),
                    tx: tx.into(),
                    ty: ty.into(), };
        (old_swap, index)

    }

    /// Creates a new random swap and replaces one of the existing one. The replaced Swap and it's
    /// index are returned as a tuple.
    pub fn mutate<R: Rng>(&mut self, rng: &mut R, ncols: T, nrows: T) -> (Swap<T>, usize) {
        let index = rng.gen::<u32>() as u64 % (nrows.into() * ncols.into());
        self.mutate_at(index as usize, rng, ncols, nrows)
    }
}
