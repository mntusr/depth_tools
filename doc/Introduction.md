# Overview

Depth tools has multiple features. However, there are some shared concepts that are often used.

The different kinds of arrays are documented at a [dedicated file](Array-formats.md). If a function returns with an array or expects one, it documents the array format by refering to its name (with some exceptions where this would limit the usability of the function). The most important formats are:

- `Im_RGB`: The array contains an RGB image.
- `Im_Depth`: The array contains a depth map. Not all pixels are meaningful, the meaningful pixels are selected using a mask.
- `Im_Mask`: The array selects some pixels from an image.

The package assumes a simple pinhole camera model. The focal length and optical center values are given in pixels. The class that describes a camera is `CameraIntrinsics`. Negative focal lengths are not supported.

Main areas:

- Loss calculation
- Depth alignment and normalization
- Point cloud creation and plotting
- Dataset handling
- Conversion between depth maps and distance maps

# Loss calculation

The following losses are supported:

- $\delta_x$ loss family
- MSE loss
- MSE loss in the log space

# Depth alignment and normalization

The package can align the shift and scale of the predicted scalar maps to minimize the MSE error.

The package can also normalize depth/distance/disparity maps.

# Point cloud operations

The following point cloud operations are supported:

- Point cloud creation for a depth map
- Create a Plotly point cloud figure from a depth map (including RGB image-based colorization)
- Create a Matplotlib point cloud figure from a depth map (including RGB image-based colorization)

# Dataset handling

The following datasets are supported:

- NYU Depth v2 (eigen split)
- A simplified and filtered version of the Hypersim dataset\* \*\*

Note that you have to download the dataset files manually, this package does not support automatic dataset download.

Alongside the datasets, the package also exposes a stable abstract protocol for the datasets.

\*: The original Hypersim dataset sometimes used a camera model that is more complex than our camera model (see [this issue](https://github.com/apple/ml-hypersim/issues/24)). We decided to filter out the affected samples. The original Hypersim dataset also contained corrupted samples ([see this issue](https://github.com/apple/ml-hypersim/issues/22)). We also filtered out those samples.

\*\*: The original Hypersim dataset contained distance maps instead of depth maps. Our loader intentionally diverges from this and produces depth maps instead. The conversion is based on [this snippet](https://github.com/apple/ml-hypersim/issues/9#issuecomment-754935697).

# Pytorch support

The Pytorch-related functions are placed into a separate package. 