// Declare the vector3 module
mod vector3;

// Bring Vector3 into scope
use vector3::Vector3;

fn main() {
    let v1: Vector3 = Vector3::new(1.0, 2.0, 3.0);
    let v2: Vector3 = Vector3::new(4.0, 5.0, 6.0);
    let sum: Vector3 =  v1.add(&v2);
    println!("Sum: {:?}", sum);
}