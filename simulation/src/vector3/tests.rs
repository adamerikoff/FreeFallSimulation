use super::Vector3;

// ==============================================
// Constructor Tests
// ==============================================

#[test]
fn test_vector3_new() {
    let v = Vector3::new(1.0, 2.0, 3.0);
    assert_eq!(v.x, 1.0, "x component should be 1.0");
    assert_eq!(v.y, 2.0, "y component should be 2.0");
    assert_eq!(v.z, 3.0, "z component should be 3.0");
}

// ==============================================
// Scalar Operations Tests
// ==============================================

#[test]
fn test_scalar_multiplication() {
    let v = Vector3::new(1.0, 2.0, 3.0);

    // Test with floating-point scalar
    let scaled = v.scalar_multiplication(2.0);
    assert_eq!(scaled.x, 2.0, "x component should be 2.0 after scaling");
    assert_eq!(scaled.y, 4.0, "y component should be 4.0 after scaling");
    assert_eq!(scaled.z, 6.0, "z component should be 6.0 after scaling");

    // Test with integer scalar
    let scaled_int = v.scalar_multiplication(2);
    assert_eq!(scaled_int.x, 2.0, "x component should be 2.0 after scaling with integer");
    assert_eq!(scaled_int.y, 4.0, "y component should be 4.0 after scaling with integer");
    assert_eq!(scaled_int.z, 6.0, "z component should be 6.0 after scaling with integer");

    // Test with zero scalar
    let scaled_zero = v.scalar_multiplication(0.0);
    assert_eq!(scaled_zero.x, 0.0, "x component should be 0.0 after scaling with zero");
    assert_eq!(scaled_zero.y, 0.0, "y component should be 0.0 after scaling with zero");
    assert_eq!(scaled_zero.z, 0.0, "z component should be 0.0 after scaling with zero");
}

#[test]
fn test_scalar_addition_edge_cases() {
    // Test with negative scalar
    let v = Vector3::new(1.0, 2.0, 3.0);
    let result = v + (-2.0);
    assert_eq!(result, Vector3::new(-1.0, 0.0, 1.0), "Adding -2.0 to (1, 2, 3) should result in (-1, 0, 1)");

    // Test with very large scalar
    let large_scalar = 1e10;
    let result_large = v + large_scalar;
    assert_eq!(result_large, Vector3::new(1e10 + 1.0, 1e10 + 2.0, 1e10 + 3.0), "Adding 1e10 to (1, 2, 3) should result in (1e10+1, 1e10+2, 1e10+3)");

    // Test with very small scalar
    let small_scalar = 1e-10;
    let result_small = v + small_scalar;
    assert_eq!(result_small, Vector3::new(1.0 + 1e-10, 2.0 + 1e-10, 3.0 + 1e-10), "Adding 1e-10 to (1, 2, 3) should result in (1+1e-10, 2+1e-10, 3+1e-10)");
}

#[test]
fn test_scalar_subtraction_edge_cases() {
    // Test with negative scalar
    let v = Vector3::new(1.0, 2.0, 3.0);
    let result = v - (-2.0);
    assert_eq!(result, Vector3::new(3.0, 4.0, 5.0), "Subtracting -2.0 from (1, 2, 3) should result in (3, 4, 5)");

    // Test with very large scalar
    let large_scalar = 1e10;
    let result_large = v - large_scalar;
    assert_eq!(result_large, Vector3::new(1.0 - 1e10, 2.0 - 1e10, 3.0 - 1e10), "Subtracting 1e10 from (1, 2, 3) should result in (1-1e10, 2-1e10, 3-1e10)");

    // Test with very small scalar
    let small_scalar = 1e-10;
    let result_small = v - small_scalar;
    assert_eq!(result_small, Vector3::new(1.0 - 1e-10, 2.0 - 1e-10, 3.0 - 1e-10), "Subtracting 1e-10 from (1, 2, 3) should result in (1-1e-10, 2-1e-10, 3-1e-10)");
}

#[test]
fn test_scalar_division() {
    let v = Vector3::new(2.0, 4.0, 6.0);

    // Test with floating-point scalar
    let divided = v / 2.0;
    assert_eq!(divided.x, 1.0, "x component should be 1.0 after division");
    assert_eq!(divided.y, 2.0, "y component should be 2.0 after division");
    assert_eq!(divided.z, 3.0, "z component should be 3.0 after division");

    // Test with integer scalar
    let divided_int = v / 2;
    assert_eq!(divided_int.x, 1.0, "x component should be 1.0 after division with integer");
    assert_eq!(divided_int.y, 2.0, "y component should be 2.0 after division with integer");
    assert_eq!(divided_int.z, 3.0, "z component should be 3.0 after division with integer");
}

// ==============================================
// Vector Operations Tests
// ==============================================

#[test]
fn test_vector_addition() {
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    let v2 = Vector3::new(4.0, 5.0, 6.0);
    let result = v1 + v2;
    assert_eq!(result, Vector3::new(5.0, 7.0, 9.0), "Adding (1, 2, 3) and (4, 5, 6) should result in (5, 7, 9)");
}

#[test]
fn test_vector_subtraction() {
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    let v2 = Vector3::new(4.0, 5.0, 6.0);
    let result = v1 - v2;
    assert_eq!(result, Vector3::new(-3.0, -3.0, -3.0), "Subtracting (4, 5, 6) from (1, 2, 3) should result in (-3, -3, -3)");
}

#[test]
fn test_vector_negation() {
    let v = Vector3::new(1.0, 2.0, 3.0);
    let negated = -v;
    assert_eq!(negated, Vector3::new(-1.0, -2.0, -3.0), "Negating (1, 2, 3) should result in (-1, -2, -3)");
}

#[test]
fn test_vector_multiplication() {
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    let v2 = Vector3::new(4.0, 5.0, 6.0);
    let result = v1 * v2;
    assert_eq!(result, Vector3::new(4.0, 10.0, 18.0), "Component-wise multiplication of (1, 2, 3) and (4, 5, 6) should result in (4, 10, 18)");
}

#[test]
fn test_vector_division() {
    let v1 = Vector3::new(4.0, 10.0, 18.0);
    let v2 = Vector3::new(2.0, 2.0, 3.0);
    let result = v1 / v2;
    assert_eq!(result, Vector3::new(2.0, 5.0, 6.0), "Component-wise division of (4, 10, 18) by (2, 2, 3) should result in (2, 5, 6)");
}

#[test]
fn test_vector_addition_edge_cases() {
    // Test with zero vector
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    let zero_vector = Vector3::new(0.0, 0.0, 0.0);
    assert_eq!(v1 + zero_vector, v1, "Adding zero vector to (1, 2, 3) should result in (1, 2, 3)");

    // Test with negative vector
    let v2 = Vector3::new(-1.0, -2.0, -3.0);
    assert_eq!(v1 + v2, Vector3::new(0.0, 0.0, 0.0), "Adding (1, 2, 3) and (-1, -2, -3) should result in (0, 0, 0)");
}

#[test]
fn test_vector_subtraction_edge_cases() {
    // Test with zero vector
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    let zero_vector = Vector3::new(0.0, 0.0, 0.0);
    assert_eq!(v1 - zero_vector, v1, "Subtracting zero vector from (1, 2, 3) should result in (1, 2, 3)");

    // Test with negative vector
    let v2 = Vector3::new(-1.0, -2.0, -3.0);
    assert_eq!(v1 - v2, Vector3::new(2.0, 4.0, 6.0), "Subtracting (-1, -2, -3) from (1, 2, 3) should result in (2, 4, 6)");
}

// ==============================================
// Advanced Vector Operations Tests
// ==============================================

#[test]
fn test_magnitude() {
    let v = Vector3::new(1.0, 2.0, 3.0);
    let magnitude = v.magnitude();
    let expected = (1.0f64.powi(2) + (2.0f64.powi(2)) + (3.0f64.powi(2))).sqrt();
    assert_eq!(magnitude, expected, "Magnitude of (1, 2, 3) should be sqrt(14)");

    // Test with zero vector
    let zero_vector = Vector3::new(0.0, 0.0, 0.0);
    assert_eq!(zero_vector.magnitude(), 0.0, "Magnitude of zero vector should be 0.0");
}

#[test]
fn test_normalize() {
    let v = Vector3::new(1.0, 2.0, 3.0);
    let normalized = v.normalize();
    let magnitude = v.magnitude();
    let expected = Vector3 {
        x: 1.0 / magnitude,
        y: 2.0 / magnitude,
        z: 3.0 / magnitude,
    };
    assert_eq!(normalized, expected, "Normalized vector should match expected result");

    // Test with zero vector (should return zero vector)
    let zero_vector = Vector3::new(0.0, 0.0, 0.0);
    assert_eq!(zero_vector.normalize(), zero_vector, "Normalizing zero vector should return zero vector");
}

#[test]
fn test_magnitude_edge_cases() {
    // Test with very large components
    let v_large = Vector3::new(1e6, 2e6, 3e6);
    let expected_magnitude = (1e6f64.powi(2) + 2e6f64.powi(2) + 3e6f64.powi(2)).sqrt();
    assert_eq!(v_large.magnitude(), expected_magnitude, "Magnitude of (1e6, 2e6, 3e6) should be sqrt(1e12 + 4e12 + 9e12)");

    // Test with very small components
    let v_small = Vector3::new(1e-6, 2e-6, 3e-6);
    let expected_magnitude_small = (1e-6f64.powi(2) + 2e-6f64.powi(2) + 3e-6f64.powi(2)).sqrt();
    assert_eq!(v_small.magnitude(), expected_magnitude_small, "Magnitude of (1e-6, 2e-6, 3e-6) should be sqrt(1e-12 + 4e-12 + 9e-12)");
}

#[test]
fn test_normalization_edge_cases() {
    let v_small = Vector3::new(1e-6, 2e-6, 3e-6);
    let normalized_small = v_small.normalize();
    let magnitude_small = v_small.magnitude();
    let expected = Vector3::new(1e-6 / magnitude_small, 2e-6 / magnitude_small, 3e-6 / magnitude_small);

    // Use a small epsilon for floating-point comparison
    let epsilon = 1e-10;
    assert!((normalized_small.x - expected.x).abs() < epsilon, "Normalization of (1e-6, 2e-6, 3e-6) should match expected x component");
    assert!((normalized_small.y - expected.y).abs() < epsilon, "Normalization of (1e-6, 2e-6, 3e-6) should match expected y component");
    assert!((normalized_small.z - expected.z).abs() < epsilon, "Normalization of (1e-6, 2e-6, 3e-6) should match expected z component");
}

#[test]
fn test_cross_product() {
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    let v2 = Vector3::new(4.0, 5.0, 6.0);
    let result = v1.cross_product(&v2);
    assert_eq!(result, Vector3::new(-3.0, 6.0, -3.0), "Cross product of (1, 2, 3) and (4, 5, 6) should result in (-3, 6, -3)");
}

#[test]
fn test_cross_product_edge_cases() {
    // Test with parallel vectors
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    let v2 = Vector3::new(2.0, 4.0, 6.0); // v2 is a scalar multiple of v1
    let result = v1.cross_product(&v2);
    assert_eq!(result, Vector3::new(0.0, 0.0, 0.0), "Cross product of parallel vectors should be zero vector");

    // Test with orthogonal vectors
    let v3 = Vector3::new(1.0, 0.0, 0.0);
    let v4 = Vector3::new(0.0, 1.0, 0.0);
    let result_orthogonal = v3.cross_product(&v4);
    assert_eq!(result_orthogonal, Vector3::new(0.0, 0.0, 1.0), "Cross product of (1, 0, 0) and (0, 1, 0) should be (0, 0, 1)");
}

#[test]
fn test_dot_product() {
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    let v2 = Vector3::new(4.0, 5.0, 6.0);
    let result = v1.dot_product(&v2);
    assert_eq!(result, 32.0, "Dot product of (1, 2, 3) and (4, 5, 6) should be 32.0");
}

#[test]
fn test_dot_product_edge_cases() {
    // Test with orthogonal vectors
    let v1 = Vector3::new(1.0, 0.0, 0.0);
    let v2 = Vector3::new(0.0, 1.0, 0.0);
    assert_eq!(v1.dot_product(&v2), 0.0, "Dot product of orthogonal vectors should be 0.0");

    // Test with parallel vectors
    let v3 = Vector3::new(1.0, 2.0, 3.0);
    let v4 = Vector3::new(2.0, 4.0, 6.0); // v4 is a scalar multiple of v3
    assert_eq!(v3.dot_product(&v4), 28.0, "Dot product of (1, 2, 3) and (2, 4, 6) should be 28.0");
}

#[test]
fn test_euclidean_distance() {
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    let v2 = Vector3::new(4.0, 5.0, 6.0);
    let result = v1.euclidean_distance(&v2);
    let expected = ((4.0f64 - 1.0f64).powi(2) + (5.0f64 - 2.0f64).powi(2) + (6.0f64 - 3.0f64).powi(2)).sqrt();
    assert_eq!(result, expected, "Euclidean distance between (1, 2, 3) and (4, 5, 6) should be sqrt(27)");
}

#[test]
fn test_euclidean_distance_edge_cases() {
    // Test with identical vectors
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    assert_eq!(v1.euclidean_distance(&v1), 0.0, "Euclidean distance between identical vectors should be 0.0");

    // Test with very large vectors
    let v2 = Vector3::new(1e6, 2e6, 3e6);
    let v3 = Vector3::new(2e6, 3e6, 4e6);
    let expected_distance = ((1e6f64 - 2e6f64).powi(2) + (2e6f64 - 3e6f64).powi(2) + (3e6f64 - 4e6f64).powi(2)).sqrt();
    assert_eq!(v2.euclidean_distance(&v3), expected_distance, "Euclidean distance between (1e6, 2e6, 3e6) and (2e6, 3e6, 4e6) should match expected result");
}

#[test]
fn test_cosine_similarity() {
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    let v2 = Vector3::new(4.0, 5.0, 6.0);
    let result = v1.cosine_similarity(&v2);
    let dot_product = v1.dot_product(&v2);
    let magnitude_v1 = v1.magnitude();
    let magnitude_v2 = v2.magnitude();
    let expected = dot_product / (magnitude_v1 * magnitude_v2);
    assert_eq!(result, expected, "Cosine similarity between (1, 2, 3) and (4, 5, 6) should match the expected value");

    // Test with zero vector
    let zero_vector = Vector3::new(0.0, 0.0, 0.0);
    assert_eq!(v1.cosine_similarity(&zero_vector), 0.0, "Cosine similarity with a zero vector should be 0.0");
}

#[test]
fn test_cosine_similarity_edge_cases() {
    // Test with identical vectors
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    assert_eq!(v1.cosine_similarity(&v1), 1.0, "Cosine similarity of a vector with itself should be 1.0");

    // Test with orthogonal vectors
    let v2 = Vector3::new(0.0, 1.0, 0.0);
    let v3 = Vector3::new(1.0, 0.0, 0.0);
    assert_eq!(v2.cosine_similarity(&v3), 0.0, "Cosine similarity of orthogonal vectors should be 0.0");

    // Test with anti-parallel vectors
    let v4 = Vector3::new(-1.0, -2.0, -3.0);
    assert_eq!(v1.cosine_similarity(&v4), -1.0, "Cosine similarity of anti-parallel vectors should be -1.0");

    // Test with very small vectors
    let v5 = Vector3::new(1e-10, 2e-10, 3e-10);
    let v6 = Vector3::new(4e-10, 5e-10, 6e-10);
    let result = v5.cosine_similarity(&v6);
    let expected = v5.dot_product(&v6) / (v5.magnitude() * v6.magnitude());
    assert!((result - expected).abs() < 1e-10, "Cosine similarity of very small vectors should match expected value");

    // Test with one zero vector
    let zero_vector = Vector3::new(0.0, 0.0, 0.0);
    assert_eq!(v1.cosine_similarity(&zero_vector), 0.0, "Cosine similarity with a zero vector should be 0.0");
}

#[test]
fn test_angle() {
    let v1 = Vector3::new(1.0, 0.0, 0.0);
    let v2 = Vector3::new(0.0, 1.0, 0.0);
    let result = v1.angle(&v2);
    let expected = std::f64::consts::FRAC_PI_2; // 90 degrees in radians
    assert!((result - expected).abs() < 1e-10, "Angle between (1, 0, 0) and (0, 1, 0) should be π/2 radians");

    // Test with parallel vectors
    let v3 = Vector3::new(2.0, 0.0, 0.0);
    assert!((v1.angle(&v3) - 0.0).abs() < 1e-10, "Angle between parallel vectors should be 0 radians");

    // Test with anti-parallel vectors
    let v4 = Vector3::new(-1.0, 0.0, 0.0);
    assert!((v1.angle(&v4) - std::f64::consts::PI).abs() < 1e-10, "Angle between anti-parallel vectors should be π radians");
}

#[test]
fn test_angle_edge_cases() {
    // Test with identical vectors
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    assert!((v1.angle(&v1) - 0.0).abs() < 1e-10, "Angle between identical vectors should be 0 radians");

    // Test with anti-parallel vectors
    let v2 = Vector3::new(-1.0, -2.0, -3.0);
    assert!((v1.angle(&v2) - std::f64::consts::PI).abs() < 1e-10, "Angle between anti-parallel vectors should be π radians");

    // Test with very small vectors
    let v3 = Vector3::new(1e-10, 2e-10, 3e-10);
    let v4 = Vector3::new(4e-10, 5e-10, 6e-10);
    let result = v3.angle(&v4);
    let expected = v3.cosine_similarity(&v4).acos();
    assert!((result - expected).abs() < 1e-10, "Angle between very small vectors should match expected value");

    // Test with one zero vector
    let zero_vector = Vector3::new(0.0, 0.0, 0.0);
    assert!((v1.angle(&zero_vector) - 0.0).abs() < 1e-10, "Angle with a zero vector should be 0 radians");
}

#[test]
fn test_projection() {
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    let v2 = Vector3::new(1.0, 0.0, 0.0);
    let result = v1.projection(&v2);
    assert_eq!(result, Vector3::new(1.0, 0.0, 0.0), "Projection of (1, 2, 3) onto (1, 0, 0) should be (1, 0, 0)");
}

#[test]
fn test_projection_edge_cases() {
    // Test with zero vector
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    let zero_vector = Vector3::new(0.0, 0.0, 0.0);
    assert_eq!(v1.projection(&zero_vector), zero_vector, "Projection onto zero vector should return zero vector");

    // Test with orthogonal vectors
    let v2 = Vector3::new(1.0, 0.0, 0.0);
    let v3 = Vector3::new(0.0, 1.0, 0.0);
    assert_eq!(v2.projection(&v3), zero_vector, "Projection of (1, 0, 0) onto (0, 1, 0) should be zero vector");

    // Test with parallel vectors
    let v4 = Vector3::new(2.0, 0.0, 0.0);
    assert_eq!(v2.projection(&v4), v2, "Projection of (1, 0, 0) onto (2, 0, 0) should be (1, 0, 0)");

    // Test with very small vectors
    let v5 = Vector3::new(1e-10, 2e-10, 3e-10);
    let v6 = Vector3::new(4e-10, 5e-10, 6e-10);
    let result = v5.projection(&v6);
    let expected = v6.scalar_multiplication(v5.dot_product(&v6) / v6.magnitude().powi(2));
    assert!((result.x - expected.x).abs() < 1e-10, "Projection of very small vectors should match expected x component");
    assert!((result.y - expected.y).abs() < 1e-10, "Projection of very small vectors should match expected y component");
    assert!((result.z - expected.z).abs() < 1e-10, "Projection of very small vectors should match expected z component");
}