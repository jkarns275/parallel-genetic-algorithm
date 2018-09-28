pub trait WeightGen {
    /// Generates a weight for a cells fitness contribution, based on position.
    fn weight(&self, col: usize, row: usize) -> f64;
}

#[derive(Copy, Clone)]
pub struct CircleWeightGen {
    pub inner_radius: f64,
    pub outer_radius: f64,
    pub position: (f64, f64),
    pub internal_weight: f64,
    pub external_weight: f64,
}

impl WeightGen for CircleWeightGen {
    fn weight(&self, col: usize, row: usize) -> f64 {
        let (x, y) = (col as f64, row as f64);
        fn distance(p0: (f64, f64), p1: (f64, f64)) -> f64 {
            let dx = p0.0 - p1.0;
            let dx2 = dx * dx;
            let dy = p0.1 - p1.1;
            let dy2 = dy * dy;
            (dx2 + dy2).sqrt()
        }

        let dist = distance((x, y), self.position);
        if dist > self.inner_radius && dist < self.outer_radius {
            self.internal_weight
        } else {
            self.external_weight
        }
    }
}

use noise::*;
use std::time::UNIX_EPOCH;

#[derive(Copy, Clone)]
pub struct PerlinWeightGen {
    pub scalex: f64,
    pub scaley: f64,
    noise: OpenSimplex
}

impl PerlinWeightGen {
    pub fn new(scalex: f64, scaley: f64) -> Self {
        let seed = UNIX_EPOCH.elapsed().unwrap().as_nanos();
        let mut noise = OpenSimplex::new().set_seed(seed as u32);
        Self {
            scalex,
            scaley,
            noise
        }
    }
}

impl WeightGen for PerlinWeightGen {
    fn weight(&self, col: usize, row: usize) -> f64 {
        self.noise.get([col as f64 * self.scalex, row as f64 * self.scaley])
    }
}