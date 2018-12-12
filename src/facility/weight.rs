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
    pub scalex: f64,
    pub scaley: f64,
}

impl WeightGen for CircleWeightGen {
    fn weight(&self, col: usize, row: usize) -> f64 {
        let (x, y) = (col as f64, row as f64);
        fn distance(scalex: f64, scaley: f64, p0: (f64, f64), p1: (f64, f64)) -> f64 {
            let dx = (p0.0 - p1.0) * scalex;
            let dx2 = dx * dx;
            let dy = (p0.1 - p1.1) * scaley;
            let dy2 = dy * dy;
            (dx2 + dy2).sqrt()
        }

        let dist = distance(self.scalex, self.scaley, (x, y), self.position);
        if dist > self.inner_radius && dist < self.outer_radius {
            self.internal_weight
        } else {
            self.external_weight
        }
    }
}
