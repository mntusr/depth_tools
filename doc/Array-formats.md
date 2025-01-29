This page details the formats of the different arrays used in this package. The array formats specify the shapes, dtypes and meanings of the different arrays.

# Im

Shape: ``(C, H, W)``, where:

* ``N``: The index of the sample.
* ``C``: The index of the channel.
* ``H``: The vertical coordiante.
* ``W``: The horizontal coordinate.

Dtype: specified by the sub-formats

The array contains an image. The pixel at the top left corner: $(\cdot, 0, 0)$.

The pixel values give the values at the center of the pixels.

## Im_RGB

Dtype: floating

Number of channels: 3

Channel order:

* ``R=0``
* ``G=1``
* ``B=2``

The array contains an RGB image.

## Im_Mask

Dtype: boolean

Number of channels: 1

The array specifies a boolean value for each pixel. The value True means that the pixel is selected by the mask.

## Im_Scalar

Dtype: floating

Number of channels: 1

The array gives a scalar value for each pixel.


### Im_FloatMask

Dtype: floating

Number of channels: 1

The array specifies a non-boolean mask for each pixel. The application of the mask on an image is the elementwise multiplication for each channel.

### Im_Depth

Dtype: floating

Number of channels: 1

The array specifies a non-boolean mask for each pixel. The application of the mask on an image is the elementwise multiplication for each channel.

### Im_VS

Dtype: floating

Number of channels: 1

The array contains the view-space position for each pixel selected by a mask.

# Scalars

Dimensions: `(N,)`, where:

* ``N``: The index of the batch.

Dtype: floating

The array contains a single scalar value for each element.

# Indices

Dimensions: `(N,)`, where:

* ``N``: The index of the batch.

Dtype: integer

The array contains indices.

# Matrix

Dimensions: `(Row, Col)`, where:

* `Row`: The row of the matrix.
* `Col`: The column of the matrix.

Dtype: floating

The array contains a matrix.

## Matrix_3x3

The array contains a 3x3 matrix. This means that the shape of the array is `(3, 3)`

## Matrix_4x4

The array contains a 3x3 matrix. This means that the shape of the array is `(3, 3)`

# MatrixI

Dimensions: `(Row, Col)`, where:

* `Row`: The row of the matrix.
* `Col`: The column of the matrix.

Dtype: integer

The array contains a matrix.

## MatrixI_3x3

The array contains a 3x3 matrix. This means that the shape of the array is `(3, 3)`