import numpy as np

# 1. Setup dimensions
rows, cols = 5, 5
np.random.seed(42)

# Initial random array
base_array = np.random.randint(0, 10, size=(rows, cols))

# 2. Initialize the Difference Array (Augmented Domain)
# We add +1 to both dimensions to accommodate the "exit" markers at index end+1
diff = np.zeros((rows + 1, cols + 1), dtype=int)

# 3. Define the Range Updates
# Each update: (row_start, col_start, row_end, col_end, value)
# Note: end indices are inclusive
updates = [
    (1, 1, 3, 3, 10),  # A 3x3 square of 10s starting at (1,1)
    (0, 2, 1, 4, 5),   # A 2x3 rectangle of 5s
    (4, 0, 4, 4, -2)   # Subtract 2 from the entire last row
]

# 4. Apply Updates using the 4-Point Marker System
# This is the discrete mixed partial derivative of the box indicator
for r1, c1, r2, c2, val in updates:
    diff[r1, c1] += val          # Start of the region
    diff[r2 + 1, c1] -= val      # Exit row boundary
    diff[r1, c2 + 1] -= val      # Exit column boundary
    diff[r2 + 1, c2 + 1] += val  # Correction for the double-subtracted corner

# 5. Reconstruct the Function (Separable Integration)
# We use NumPy's cumsum (cumulative sum) which is the prefix sum operator S
# First pass: Integrate along rows (Axis 0)
integrated = diff.cumsum(axis=0)
# Second pass: Integrate along columns (Axis 1)
integrated = integrated.cumsum(axis=1)

# 6. Final Result
# We slice back to the original (rows, cols) to remove the augmented padding
final_contribution = integrated[:rows, :cols]
result = base_array + final_contribution

print("Base Array:\n", base_array)
print("\nFinal Array after Range Updates:\n", result)

# Verification: Compare against a naive slow-loop approach
naive = base_array.copy()
for r1, c1, r2, c2, val in updates:
    naive[r1:r2+1, c1:c2+1] += val

assert np.array_equal(result, naive)
print("\nVerification successful: Difference array matches naive implementation.")
