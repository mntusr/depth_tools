from typing import Any

import numpy as np


def masked_median_unchecked(
    a: np.ndarray,
    mask: np.ndarray,
    along_all_dims_except_0: bool,
    keep_dropped_dims: bool = False,
) -> np.ndarray:
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
        Same as the ``keepdims`` argument of the similar funcitons.

    Returns
    -------
    v
        The calculated median. Format:

        - a single 0-dimensional scalar if ``along_all_dims_except_0=False`` and ``keep_dropped_dims=False``
        - a one-dimensional array if ``along_all_dims_except_0=True`` and ``keep_dropped_dims=False``, where each element denotes the corresponding median
        - an array with the same number of dimensions as ``a`` where dimension 0 denotes the index of the element, the other dimensions have length 1, if ``along_all_dims_except_0=<any>`` and ``keep_dropped_dims=True``, where each element denotes the corresponding median
    """

    if along_all_dims_except_0:
        mean_val_list: list[np.ndarray] = []

        # For loops are generally not efficient.
        # However in this case, I was not able to come up
        # with any other idea that plays nicely with nans and
        # simple to implement.
        #
        # The first dimension is typically not that large (1-32) anyway
        # (at least for the cases where this function is used).
        for i in range(len(a)):
            mean_val = np.median(a[i][mask[i]])
            mean_val_list.append(mean_val)
        mean_val_arr = np.array(mean_val_list)
        if keep_dropped_dims:
            new_shape: list[int] = [len(mean_val_arr)] + [1] * (len(a.shape) - 1)
            return np.reshape(mean_val_arr, new_shape)
        else:
            return mean_val_arr
    else:
        a = a.copy()
        a[~mask] = np.nan
        nan_canary = a[mask].min() * 0
        return np.array(np.nanmedian(a, keepdims=keep_dropped_dims)) + nan_canary


def masked_mean_unchecked(
    a: np.ndarray,
    mask: np.ndarray,
    along_all_dims_except_0: bool,
    keep_dropped_dims: bool = False,
) -> np.ndarray:
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
        Same as the ``keepdims`` argument of the similar funcitons.

    Returns
    -------
    v
        The calculated mean. Format:

        - a single 0-dimensional scalar if ``along_all_dims_except_0=False`` and ``keep_dropped_dims=False``
        - a one-dimensional array if ``along_all_dims_except_0=True`` and ``keep_dropped_dims=False``, where each element denotes the corresponding mean
        - an array with the same number of dimensions as ``a`` where dimension 0 denotes the index of the element, the other dimensions have length 1, if ``along_all_dims_except_0=<any>`` and ``keep_dropped_dims=True``, where each element denotes the corresponding mean
    """

    if along_all_dims_except_0:
        mean_val_list: list[np.ndarray] = []

        # For loops are generally not efficient.
        # However in this case, I was not able to come up
        # with any other idea that plays nicely with nans and
        # simple to implement.
        #
        # The first dimension is typically not that large (1-32) anyway
        # (at least for the cases where this function is used).
        for i in range(len(a)):
            mean_val = np.mean(a[i], where=mask[i])
            mean_val_list.append(mean_val)
        mean_val_arr = np.array(mean_val_list)
        if keep_dropped_dims:
            new_shape: list[int] = [len(mean_val_arr)] + [1] * (len(a.shape) - 1)
            return np.reshape(mean_val_arr, new_shape)
        else:
            return mean_val_arr
    else:
        a = a.copy()
        a[~mask] = np.nan
        nan_canary = a[mask].min() * 0
        return np.array(np.nanmean(a, keepdims=keep_dropped_dims)) + nan_canary


def new_full(
    like: np.ndarray,
    value: Any,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
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
    return np.full_like(like, value, shape=shape)
