from typing import Any

import torch


def masked_median_unchecked(
    a: torch.Tensor,
    mask: torch.Tensor,
    along_all_dims_except_0: bool,
    keep_dropped_dims: bool = False,
) -> torch.Tensor:
    """
    A function that calculates the median for the array elements selected by a given mask.

    This function does not check its arguments.

    Parameters
    ----------
    a
        The array on which the median calculation should be done. Format: any floating point array
    mask
        The array that selects the elements used to calculate the median. Format: its shape is the same as the shape of ``a``, its dtype is bool
    along_all_dims_except_0
        If true, then the median calculation is done along all dimensions, except 0. Otherwise, the median calculation is done alongside all dimensions.
    keep_dropped_dims
        Same as the ``keepdim`` argument of the similar funcitons.

    Returns
    -------
    v
        The calculated median. Format:

        - a single 0-dimensional scalar if ``along_all_dims_except_0=False`` and ``keep_dropped_dims=False``
        - a one-dimensional array if ``along_all_dims_except_0=True`` and ``keep_dropped_dims=False``, where each element denotes the corresponding median
        - an array with the same number of dimensions as ``a`` where dimension 0 denotes the index of the element, the other dimensions have length 1, if ``along_all_dims_except_0=<any>`` and ``keep_dropped_dims=True``, where each element denotes the corresponding median
    """

    if along_all_dims_except_0:
        mean_val_list: list[torch.Tensor] = []

        # For loops are generally not efficient.
        # However in this case, I was not able to come up
        # with any other idea that plays nicely with nans and
        # simple to implement.
        #
        # The first dimension is typically not that large (1-32) anyway
        # (at least for the cases where this function is used).
        for i in range(len(a)):
            mean_val = torch.median(a[i][mask[i]])
            mean_val_list.append(mean_val)
        mean_val_arr = torch.stack(mean_val_list)
        if keep_dropped_dims:
            new_shape: list[int] = [len(mean_val_arr)] + [1] * (len(a.shape) - 1)
            return torch.reshape(mean_val_arr, new_shape)
        else:
            return mean_val_arr
    else:
        median_arr = torch.median(a[mask])

        if keep_dropped_dims:
            new_shape = [1] * len(a.shape)
            return torch.reshape(median_arr, new_shape)
        else:
            return median_arr


def masked_mean_unchecked(
    a: torch.Tensor,
    mask: torch.Tensor,
    along_all_dims_except_0: bool,
    keep_dropped_dims: bool = False,
) -> torch.Tensor:
    """
    A function that calculates the mean for the array elements selected by a given mask.

    This function does not check its arguments.

    Parameters
    ----------
    a
        The array on which the mean calculation should be done. Format: any floating point array
    mask
        The array that selects the elements used to calculate the median. Format: its shape is the same as the shape of ``a``, its dtype is bool
    along_all_dims_except_0
        If true, then the mean calculation is done along all dimensions, except 0. Otherwise, the mean calculation is done alongside all dimensions.
    keep_dropped_dims
        Same as the ``keepdim`` argument of the similar funcitons.

    Returns
    -------
    v
        The calculated median. Format:

        - a single 0-dimensional scalar if ``along_all_dims_except_0=False`` and ``keep_dropped_dims=False``
        - a one-dimensional array if ``along_all_dims_except_0=True`` and ``keep_dropped_dims=False``, where each element denotes the corresponding mean
        - an array with the same number of dimensions as ``a`` where dimension 0 denotes the index of the element, the other dimensions have length 1, if ``along_all_dims_except_0=<any>`` and ``keep_dropped_dims=True``, where each element denotes the corresponding mean
    """

    if along_all_dims_except_0:
        mean_val_list: list[torch.Tensor] = []

        # For loops are generally not efficient.
        # However in this case, I was not able to come up
        # with any other idea that plays nicely with nans and
        # simple to implement.
        #
        # The first dimension is typically not that large (1-32) anyway
        # (at least for the cases where this function is used).
        for i in range(len(a)):
            mean_val = torch.mean(a[i][mask[i]])
            mean_val_list.append(mean_val)
        mean_val_arr = torch.stack(mean_val_list)
        if keep_dropped_dims:
            new_shape: list[int] = [len(mean_val_arr)] + [1] * (len(a.shape) - 1)
            return torch.reshape(mean_val_arr, new_shape)
        else:
            return mean_val_arr
    else:
        mean_val = torch.mean(a[mask])
        if keep_dropped_dims:
            new_shape = [1] * len(a.shape)
            return torch.reshape(mean_val, new_shape)
        else:
            return mean_val


def new_full(
    like: torch.Tensor,
    value: Any,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """
    Implements a ``full_like`` operation that plays nicely with tensor subclassing in Pytorch.

    Parameters
    ----------
    like
        The original array. Format: any array.
    value
        The value.
    shape
        The shape of the created array.
    """
    if shape is None:
        real_shape = value.shape
    else:
        real_shape = shape

    return like.new_full(real_shape, value)


def new_zeros(
    like: torch.Tensor,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """
    Implements a ``zeros_like`` operation that plays nicely with tensor subclassing in Pytorch.

    Parameters
    ----------
    like
        The original array. Format: any array.
    shape
        The shape of the created array.
    """
    if shape is None:
        real_shape = like.shape
    else:
        real_shape = shape

    return like.new_zeros(size=real_shape, dtype=like.dtype)
