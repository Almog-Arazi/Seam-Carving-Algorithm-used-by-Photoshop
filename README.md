# Seam Carving Algorithm - Content-Aware Image Resizing

A Python implementation of the Seam Carving algorithm, the revolutionary content-aware image resizing technique used in Adobe Photoshop's "Content-Aware Scale" feature. This project demonstrates how images can be intelligently resized while preserving important visual content and semantic features.

## Table of Contents
- [Overview](#overview)
- [Algorithm Description](#algorithm-description)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Performance Analysis](#performance-analysis)
- [Usage](#usage)
- [Requirements](#requirements)
- [Authors](#authors)

## Overview

Traditional image resizing methods like scaling or cropping either distort the image or remove important content. Seam Carving, introduced by Avidan and Shamir in their seminal 2007 paper, offers a content-aware alternative that intelligently removes or adds pixels based on their importance to the overall image composition.

This implementation provides two approaches to seam carving:
1. **Greedy Algorithm** - Fast, locally optimal seam selection
2. **Dynamic Programming Algorithm** - Slower but globally optimal seam selection

## Algorithm Description

### Core Concept

Seam Carving works by identifying and removing (or duplicating) "seams" - connected paths of pixels that have minimal energy. A seam is a path of pixels connected from top to bottom (vertical seam) or left to right (horizontal seam), where each pixel is adjacent to its neighbors.

### Step-by-Step Process

#### 1. Energy Calculation

The algorithm begins by computing an energy map that identifies important features in the image. The energy function is based on gradient magnitude:

```
E(I) = |∂I/∂x| + |∂I/∂y|
```

Where high gradient values indicate important edges and features that should be preserved.

#### 2. Forward-Looking Cost

To avoid visual artifacts, the implementation uses forward-looking energy, which considers the cost of newly created edges when a seam is removed:

- **C_L**: Cost of moving up-left
- **C_V**: Cost of moving up-vertically  
- **C_R**: Cost of moving up-right

These costs incorporate the difference between neighboring pixels that will become adjacent after seam removal.

#### 3. Seam Finding

**Greedy Approach:**
- Start with the minimum energy pixel in the first row
- For each subsequent row, choose the neighbor (up-left, up, or up-right) with minimum total cost
- Time Complexity: O(w + h)

**Dynamic Programming Approach:**
- Build a cumulative cost matrix M from top to bottom
- M[i,j] = E[i,j] + min(M[i-1,j-1] + C_L, M[i-1,j] + C_V, M[i-1,j+1] + C_R)
- Backtrack from the minimum value in the bottom row
- Time Complexity: O(w × h)

#### 4. Seam Removal

Once the optimal seam is identified, remove the pixels along the seam path and shift the remaining pixels to fill the gap. Repeat this process for the desired number of seams.

### Vertical and Horizontal Seam Removal

The implementation supports both vertical and horizontal seam removal:
- **Vertical seams**: Remove directly by finding paths from top to bottom
- **Horizontal seams**: Rotate the image 90°, remove vertical seams, then rotate back

## Implementation Details

### Class Architecture

```
SeamImage (Base Class)
│
├── Image loading and preprocessing
├── Grayscale conversion
├── Gradient magnitude calculation
├── Forward-looking cost calculation
├── Seam removal infrastructure
└── Image rotation utilities
    │
    ├── GreedySeamImage (Subclass)
    │   └── Greedy seam finding algorithm
    │
    └── DPSeamImage (Subclass)
        ├── Dynamic programming cost matrix
        └── Backtracking matrix
```

### Key Methods

**Base Class (`SeamImage`):**
- `rgb_to_grayscale()`: Converts RGB to weighted grayscale
- `calc_gradient_magnitude()`: Computes energy map using gradient
- `calc_C()`: Calculates forward-looking cost matrices (C_L, C_V, C_R)
- `remove_seam()`: Removes a seam from the image
- `rotate_mats()`: Rotates all matrices for horizontal seam processing
- `seams_removal_vertical()`: Removes vertical seams
- `seams_removal_horizontal()`: Removes horizontal seams

**GreedySeamImage:**
- `find_minimal_seam()`: Greedy row-by-row seam selection

**DPSeamImage:**
- `calc_M()`: Builds dynamic programming cost matrix
- `find_minimal_seam()`: Optimal seam selection via backtracking

### Optimization

The implementation leverages several optimization techniques:
- NumPy vectorized operations for matrix computations
- Numba JIT compilation for performance-critical sections
- Efficient matrix updates during seam removal
- Index mapping to track original pixel positions

## Results

### Example 1: Koala Image Resizing

**Original Image:**
The test image features a koala sitting in a tree with background foliage.

**Seam Removal:**
- 150 vertical seams removed
- 50 horizontal seams removed

**Greedy vs Dynamic Programming Comparison:**

The greedy algorithm, while faster, makes locally optimal decisions that can lead to artifacts:
- May cut through important objects (e.g., the koala's back, tree trunk)
- Creates visible distortions when encountering uniform regions

The dynamic programming algorithm produces superior results:
- Better preservation of the koala's shape
- Smoother removal of background content
- More natural-looking resized image

### Example 2: Landscape Image Comparison

**Original Image:** Beach landscape (Palawan) - 295 × 886 pixels

**Comparison with Bilinear Interpolation:**

| Method | Scale Factor 0.6×1.0 | Scale Factor 1.0×0.6 |
|--------|---------------------|---------------------|
| **Bilinear** | Uniform vertical compression, distorts all content equally | Uniform horizontal compression, aspect ratio affected |
| **Seam Carving** | Intelligently removes less important regions, preserves main subjects | Content-aware reduction, maintains object proportions |

**Key Observation:** Seam Carving preserves the geometry of important objects while traditional rescaling modifies all pixels uniformly, often distorting important content.

### Seam Visualization

The implementation includes visualization capabilities that display removed seams in red on the original image. This reveals interesting patterns:
- Seams tend to avoid high-contrast edges
- Paths naturally flow through low-energy regions
- Seams rarely cross important objects directly

## Performance Analysis

### Runtime Comparison (Koala Image)

**Greedy Algorithm:**
- 150 vertical seams: 2.08 seconds (72.46 seams/sec)
- 50 horizontal seams: 0.467 seconds (108.70 seams/sec)

**Dynamic Programming Algorithm:**
- 150 vertical seams: 7.64 seconds (19.65 seams/sec)
- 50 horizontal seams: 2.26 seconds (22.22 seams/sec)

### Time Complexity

**Greedy:**
- Per seam: O(w + h) where w = width, h = height
- For n seams: O(n × (w + h))
- Faster but produces suboptimal results

**Dynamic Programming:**
- Per seam: O(w × h) for matrix construction + O(h) for backtracking
- For n seams: O(n × w × h)
- Slower but produces globally optimal seams

### Trade-offs

| Aspect | Greedy | Dynamic Programming |
|--------|--------|-------------------|
| **Speed** | ~3-4× faster | Slower |
| **Quality** | Can create artifacts | Superior visual quality |
| **Memory** | Minimal overhead | Requires O(w×h) for DP matrix |
| **Use Case** | Quick previews, less critical applications | Final production, high-quality results |

## Usage

### Basic Usage

```python
from utils import GreedySeamImage, DPSeamImage

# Using Greedy Algorithm
img = GreedySeamImage('path/to/image.jpg', vis_seams=True)
img.seams_removal_vertical(100)  # Remove 100 vertical seams
img.seams_removal_horizontal(50)  # Remove 50 horizontal seams

# Using Dynamic Programming
img = DPSeamImage('path/to/image.jpg', vis_seams=True)
img.seams_removal_vertical(100)
img.seams_removal_horizontal(50)
```

### Resizing to Specific Dimensions

```python
from utils import DPSeamImage, scale_to_shape, resize_seam_carving

img_path = 'images/koala.jpg'
seam_img = DPSeamImage(img_path)

# Define scale factors [height_factor, width_factor]
scale_factors = [0.8, 0.6]  # Reduce to 80% height, 60% width

# Calculate target shape
orig_shape = seam_img.rgb.shape[:2]
new_shape = scale_to_shape(orig_shape, scale_factors)

# Resize using seam carving
resized = resize_seam_carving(seam_img, (orig_shape, new_shape))
```

### Visualization

```python
import matplotlib.pyplot as plt

# Show original, seams, and resized images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img.rgb)
axes[0].set_title('Original')
axes[1].imshow(img.seams_rgb)
axes[1].set_title('Seam Visualization')
axes[2].imshow(img.resized_rgb)
axes[2].set_title('Resized')
plt.show()
```

## Requirements

```
numpy
matplotlib
PIL (Pillow)
numba
tqdm
```

Install dependencies:
```bash
pip install numpy matplotlib pillow numba tqdm
```

## Technical Details

### Forward Energy vs Backward Energy

This implementation uses **forward energy** calculation, which considers the cost of newly created edges when a seam is removed. This produces better results than backward energy (original paper) by reducing artifacts in areas with strong edges.

### Image Processing Pipeline

1. Load RGB image
2. Convert to weighted grayscale (0.299R + 0.587G + 0.114B)
3. Calculate gradient magnitude for energy map
4. Compute forward-looking costs
5. Find optimal seam (greedy or DP)
6. Remove seam and update all matrices
7. Repeat steps 3-6 for desired number of seams
8. Output resized image

### Boundary Handling

- Edges are padded with 0.5 to prevent outliers
- Seams cannot wrap around image boundaries
- Infinite cost assigned to invalid neighbor positions

## Limitations and Future Work

### Current Limitations
- Cannot preserve specific objects (manual masking not implemented)
- Seam removal only (addition implemented as bonus but not fully tested)
- Sequential processing (parallel seam removal not supported)
- Limited to RGB images

### Potential Extensions
- **Object Removal**: Mark objects for priority removal
- **Object Protection**: Mask important regions with high energy
- **Multi-size Generation**: Pre-compute multiple resolutions
- **Video Seam Carving**: Extend to video with temporal coherence
- **GPU Acceleration**: CUDA implementation for real-time processing

## References

- Avidan, S., & Shamir, A. (2007). Seam carving for content-aware image resizing. ACM Transactions on Graphics (TOG), 26(3), 10.
- [Original Paper](https://faculty.runi.ac.il/arik/scweb/imret/imret.pdf)



Computer Graphics Course Project

