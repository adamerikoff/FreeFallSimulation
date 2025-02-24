// Re-export the Vector3 struct from vector3.rs
pub use self::vector3::Vector3;

// Declare the vector3 submodule
mod vector3;

// Include the tests module when running tests
#[cfg(test)]
mod tests;