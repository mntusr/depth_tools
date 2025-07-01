The Pytorch implementation of some functions are generated automatically from the Numpy implementation by `generate_pytorch.py`. This generator uses string replacement to replace the Numpy functions to the equivalent Pytorch ones in the following files:

- `depth_tools/_losses.py`
- `depth_tools/_align_depth.py`
- `depth_tools/_normalize_values.py`

This means that the Numpy implementation should specially written in a way that makes this automatic conversion possible. In other words: 1) The Numpy implementation should consider the gradient calculation of Pytorch too. 2) If a Numpy function behaves slightly differently than the Pytorch equivalent, then this function should be abstracted away into to non-converted files that provide framework-agnostic behavior.
