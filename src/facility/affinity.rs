use rand::Rng;

pub trait Affinity where Self: Sized {
    fn empty() -> Self;
    fn aff_impl(&self, other: &Self) -> f64;
    fn is_empty(&self) -> bool;
    fn affinity(&self, other: &Self) -> f64 {
        if self.is_empty() || other.is_empty() {
            0.0
        } else {
            f64::abs(self.aff_impl(other))
        }
    }
    fn zero() -> Self;
    fn magnitude(&self) -> f64 {
        let zero = Self::zero();
        self.affinity(&zero)
    }
    fn max() -> f64;
    fn gen<R: Rng>(rng: &mut R) -> Self;
}

impl Affinity for f64 {
    fn empty() -> Self { -1.0 }

    fn aff_impl(&self, other: &Self) -> f64 { self - other }

    fn is_empty(&self) -> bool { *self < 0.0 }

    fn zero() -> Self { 0.0 }

    fn max() -> f64 { 1.0 }

    fn gen<R: Rng>(rng: &mut R) -> Self {
        rng.gen()
    }
}

impl Affinity for (f64, f64) {
    fn empty() -> Self { (-1.0, -1.0) }

    fn aff_impl(&self, other: &Self) -> f64 {
        let dx = other.0 - self.0;
        let dy = other.1 - self.1;
        (dy*dx + dy*dy).abs().sqrt()
    }

    fn is_empty(&self) -> bool { self.0.is_empty() }

    fn zero() -> Self { (0.0, 0.0) }

    fn max() -> f64 { 2.0f64.sqrt() }

    fn gen<R: Rng>(rng: &mut R) -> Self { rng.gen() }
}

impl Affinity for (f64, f64, f64) {
    fn empty() -> Self { (-1.0, -1.0, -1.0) }

    fn aff_impl(&self, other: &Self) -> f64 {
        let dx = other.0 - self.0;
        let dy = other.1 - self.1;
        let dz = other.2 - self.2;
        (dy*dx + dy*dy + dz*dz).abs().sqrt()
    }

    fn is_empty(&self) -> bool { self.0.is_empty() }

    fn zero() -> Self { (0.0, 0.0, 0.0) }

    fn max() -> f64 { 3.0f64.sqrt() }

    fn gen<R: Rng>(rng: &mut R) -> Self { rng.gen() }
}

impl Affinity for (f64, f64, f64, f64) {
    fn empty() -> Self { (-1.0, -1.0, -1.0, -1.0) }

    fn aff_impl(&self, other: &Self) -> f64 {
        let dx = other.0 - self.0;
        let dy = other.1 - self.1;
        let dz = other.2 - self.2;
        let dw = other.3 - self.3;
        (dy*dx + dy*dy + dz*dz + dw*dw).abs().sqrt()
    }

    fn is_empty(&self) -> bool { self.0.is_empty() }

    fn zero() -> Self { (0.0, 0.0, 0.0, 0.0) }

    fn max() -> f64 { 4.0f64.sqrt() }

    fn gen<R: Rng>(rng: &mut R) -> Self { rng.gen() }
}

impl Affinity for i32 {
    fn empty() -> Self { -1 }

    fn aff_impl(&self, other: &Self) -> f64 { (other - self) as f64 }

    fn is_empty(&self) -> bool { *self < 0 }

    fn max() -> f64 { 1.0 }
    fn zero() -> Self { 0 }

    fn gen<R: Rng>(rng: &mut R) -> Self {
        match rng.gen::<u8>() & 7 {
            0...5 => 1,
            6...7 => 0,
            _ => 0
        }
    }
}