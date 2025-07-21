import numpy as np
from scipy.stats import entropy
from collections import Counter
import math

def calculate_image_entropy_scipy(binary_image):
    """
    Calculate entropy of a binarized image using SciPy's entropy function.

    Parameters:
    binary_image (numpy.ndarray): Binary image as NumPy array (0s and 1s)

    Returns:
    float: Entropy value in bits
    """
    # Flatten the image to 1D array
    flat_image = binary_image.flatten()

    # Count occurrences of each pixel value
    value_counts = np.bincount(flat_image)

    # Remove zero counts to avoid issues with probability calculation
    probabilities = value_counts[value_counts > 0] / len(flat_image)

    # Calculate entropy using SciPy (returns in nats by default)
    entropy_nats = entropy(probabilities)

    # Convert from nats to bits (divide by ln(2))
    entropy_bits = entropy_nats / np.log(2)

    return entropy_bits

def calculate_image_entropy_manual(binary_image):
    """
    Calculate entropy of a binarized image using manual implementation.

    Parameters:
    binary_image (numpy.ndarray): Binary image as NumPy array (0s and 1s)

    Returns:
    float: Entropy value in bits
    """
    # Flatten the image to 1D array
    flat_image = binary_image.flatten()

    # Count occurrences of each unique pixel value
    unique_values, counts = np.unique(flat_image, return_counts=True)

    # Calculate probabilities
    total_pixels = len(flat_image)
    probabilities = counts / total_pixels

    # Calculate entropy manually using Shannon's formula: H = -Σ(p * log2(p))
    entropy_value = 0.0
    for prob in probabilities:
        if prob > 0:  # Avoid log(0)
            entropy_value -= prob * math.log2(prob)

    return entropy_value

def calculate_image_entropy_optimized(binary_image):
    """
    Optimized entropy calculation specifically for binary images.

    Parameters:
    binary_image (numpy.ndarray): Binary image as NumPy array (0s and 1s)

    Returns:
    float: Entropy value in bits
    """
    # For binary images, we only need to count 1s (or 0s)
    total_pixels = binary_image.size
    ones_count = np.sum(binary_image)
    zeros_count = total_pixels - ones_count

    # Handle edge cases where all pixels are the same
    if ones_count == 0 or zeros_count == 0:
        return 0.0

    # Calculate probabilities
    p1 = ones_count / total_pixels
    p0 = zeros_count / total_pixels

    # Calculate binary entropy: H = -p0*log2(p0) - p1*log2(p1)
    entropy_value = -(p0 * math.log2(p0) + p1 * math.log2(p1))

    return entropy_value

# Example usage and testing
if __name__ == "__main__":
    # Create test binary images

    # Test 1: Random binary image
    np.random.seed(42)
    random_binary = np.random.randint(0, 2, size=(100, 100))

    # Test 2: All zeros (minimum entropy)
    all_zeros = np.zeros((50, 50), dtype=int)

    # Test 3: All ones (minimum entropy)
    all_ones = np.ones((50, 50), dtype=int)

    # Test 4: Checkerboard pattern (maximum entropy for binary)
    checkerboard = np.zeros((50, 50), dtype=int)
    checkerboard[::2, ::2] = 1
    checkerboard[1::2, 1::2] = 1

    test_images = {
        "Random binary": random_binary,
        "All zeros": all_zeros,
        "All ones": all_ones,
        "Checkerboard": checkerboard
    }

    print("Image Entropy Calculations:")
    print("=" * 50)

    for name, image in test_images.items():
        entropy_scipy = calculate_image_entropy_scipy(image)
        entropy_manual = calculate_image_entropy_manual(image)
        entropy_optimized = calculate_image_entropy_optimized(image)

        print(f"\n{name}:")
        print(f"  SciPy method:     {entropy_scipy:.6f} bits")
        print(f"  Manual method:    {entropy_manual:.6f} bits")
        print(f"  Optimized method: {entropy_optimized:.6f} bits")

        # Verify all methods give same result
        assert abs(entropy_scipy - entropy_manual) < 1e-10
        assert abs(entropy_scipy - entropy_optimized) < 1e-10

    print(f"\n{'='*50}")
    print("All methods produce identical results!")
    print("\nNote: Maximum entropy for binary images is 1.0 bit")
    print("      (achieved when p(0) = p(1) = 0.5)")