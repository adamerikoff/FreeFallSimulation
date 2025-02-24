use std::ops::{Add, Sub, Mul, Div, Neg};

/// Represents a 3D vector with x, y, and z components.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

// ==============================================
// Operator Overloading Implementations
// ==============================================

/// Vector3 + Scalar (adds scalar to each component)
impl<T> Add<T> for Vector3
where
    T: Into<f64>,
{
    type Output = Self;

    fn add(self, scalar: T) -> Self {
        let scalar = scalar.into();
        Vector3 {
            x: self.x + scalar,
            y: self.y + scalar,
            z: self.z + scalar,
        }
    }
}

/// Vector3 - Scalar (subtracts scalar from each component)
impl<T> Sub<T> for Vector3
where 
    T: Into<f64>,
{
    type Output = Self;
    
    fn sub(self, scalar: T) -> Self::Output {
        let scalar = scalar.into();
        Vector3 {
            x: self.x - scalar,
            y: self.y - scalar,
            z: self.z - scalar,
        }
    }
}

/// Vector3 + Vector3 (component-wise addition)
impl Add for Vector3 {
    type Output = Self;
    
    fn add(self, other: Self) -> Self::Output {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

/// Vector3 - Vector3 (component-wise subtraction)
impl Sub for Vector3 {
    type Output = Self;
    
    fn sub(self, other: Self) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

/// Negation of Vector3 (multiplies each component by -1)
impl Neg for Vector3 {
    type Output = Self;

    fn neg(self) -> Self {
        Vector3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Vector3 * Scalar (component-wise multiplication)
impl<T> Mul<T> for Vector3 
where 
    T: Into<f64>,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self::Output {
        let scalar = scalar.into();
        Vector3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

/// Vector3 / Scalar (component-wise division)
impl<T> Div<T> for Vector3 
where 
    T: Into<f64>,
{
    type Output = Self;

    fn div(self, scalar: T) -> Self::Output {
        let scalar = scalar.into();
        if scalar == 0.0 {
            panic!("Division by zero");
        }
        Vector3 {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

/// Vector3 * Vector3 (component-wise multiplication)
impl Mul for Vector3 {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Vector3 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

/// Vector3 / Vector3 (component-wise division)
impl Div for Vector3 {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        if other.x == 0.0 || other.y == 0.0 || other.z == 0.0 {
            panic!("Division by zero");
        }
        Vector3 {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
        }
    }
}

// ==============================================
// Vector3 Methods
// ==============================================

impl Vector3 {
    /// Creates a new Vector3 with the given x, y, and z components.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vector3 { x, y, z }
    }

    /// Multiplies the vector by a scalar.
    pub fn scalar_multiplication(&self, scalar: impl Into<f64>) -> Self {
        let scalar = scalar.into();
        Vector3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }

    /// Calculates the magnitude (Euclidean norm) of the vector.
    pub fn magnitude(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }

    /// Normalizes the vector (returns a unit vector in the same direction).
    pub fn normalize(&self) -> Vector3 {
        let magnitude = self.magnitude();
        if magnitude == 0.0 {
            Vector3::new(0.0, 0.0, 0.0) // Return zero vector to avoid division by zero
        } else {
            self.scalar_multiplication(1.0 / magnitude)
        }
    }

    /// Negates the vector (returns a vector with all components multiplied by -1).
    pub fn negate(&self) -> Vector3 {
        self.scalar_multiplication(-1)
    }

    /// Returns a new vector with the absolute values of each component.
    pub fn absolute_values(&self) -> Vector3 {
        Vector3 {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    /// Sums all elements of the vector.
    pub fn sum(&self) -> f64 {
        self.x + self.y + self.z
    }

    /// Adds two vectors together.
    pub fn add(&self, other: &Vector3) -> Self {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Subtracts one vector from another.
    pub fn subtract(&self, other: &Vector3) -> Self {
        self.add(&other.negate())
    }

    /// Computes the cross product of two vectors.
    pub fn cross_product(&self, other: &Vector3) -> Vector3 {
        Vector3 { 
            x: (self.y * other.z - self.z * other.y),
            y: (self.z * other.x - self.x * other.z),
            z: (self.x * other.y - self.y * other.x),
        }
    }

    /// Computes the Hadamard product (component-wise multiplication) of two vectors.
    pub fn hadamard_product(&self, other: &Vector3) -> Self {
        Vector3 {
            x: (self.x * other.x),
            y: (self.y * other.y),
            z: (self.z * other.z),
        }
    }

    /// Computes the dot product of two vectors.
    pub fn dot_product(&self, other: &Vector3) -> f64 {
        (self.hadamard_product(other)).sum()
    }

    /// Computes the Euclidean distance between two vectors.
    pub fn euclidean_distance(&self, other: &Vector3) -> f64 {
        (self.subtract(other)).magnitude()
    }

    /// Computes the cosine similarity between two vectors.
    pub fn cosine_similarity(&self, other: &Vector3) -> f64 {
        let a_magnitude = self.magnitude();
        let b_magnitude = other.magnitude();
        if a_magnitude == 0.0 || b_magnitude == 0.0 {
            return 0.0; // Return 0 if either vector is a zero vector
        }
        let dot_product = self.dot_product(other);
        let cosine_sim = dot_product / (a_magnitude * b_magnitude);
        cosine_sim.clamp(-1.0, 1.0) // Clamp to avoid NaN
    }

    /// Computes the angle (in radians) between two vectors.
    pub fn angle(&self, other: &Vector3) -> f64 {
        if self.magnitude() == 0.0 || other.magnitude() == 0.0 {
            return 0.0; // Return 0 if either vector is a zero vector
        }
        let cosine_sim = self.cosine_similarity(other);
        cosine_sim.acos()
    }

    /// Computes the projection of this vector onto another vector.
    pub fn projection(&self, other: &Vector3) -> Vector3 {
        let other_magnitude_squared = other.magnitude().powi(2);
        if other_magnitude_squared == 0.0 {
            return Vector3::new(0.0, 0.0, 0.0); // Return zero vector if other is zero
        }
        other.scalar_multiplication(self.dot_product(other) / other_magnitude_squared)
    }
    
}