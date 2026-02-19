from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from npy_unittest import NpyTestCase

import depth_tools._diverging_functions_internal
import depth_tools.pt._diverging_functions_internal


class TestImportantDivergingFunctions(NpyTestCase):
    def test_masked_mean__single__nokeepdim(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.expand_dims(im, 0)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_mean = depth_tools._diverging_functions_internal.masked_mean_unchecked(
            a=im, along_all_dims_except_0=False, keep_dropped_dims=False, mask=mask
        )
        expected_mean = im.mean(where=mask)
        self.assertFalse(np.isnan(expected_mean))  # self test

        self.assertAllclose(actual_mean, expected_mean)
        self.assertEqual(len(actual_mean.shape), 0)

    def test_masked_mean__single__nokeepdim__nan(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.expand_dims(im, 0)
        im[:, -1, -1] = np.nan

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_mean = depth_tools._diverging_functions_internal.masked_mean_unchecked(
            a=im, along_all_dims_except_0=False, keep_dropped_dims=False, mask=mask
        )
        expected_mean = im.mean(where=mask)
        self.assertTrue(np.isnan(expected_mean))  # self test

        self.assertAllclose(actual_mean, expected_mean, equal_nan=True)
        self.assertEqual(len(actual_mean.shape), 0)

    def test_masked_mean__single__keepdim(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.expand_dims(im, 0)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_mean = depth_tools._diverging_functions_internal.masked_mean_unchecked(
            a=im, along_all_dims_except_0=False, keep_dropped_dims=True, mask=mask
        )
        expected_mean = im.mean(where=mask, keepdims=True)
        self.assertFalse(np.isnan(expected_mean))  # self test

        self.assertAllclose(actual_mean, expected_mean)
        self.assertEqual(len(actual_mean.shape), len(im.shape))

    def test_masked_mean__single__keepdim__nan(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.expand_dims(im, 0)
        im[:, -1, -1] = np.nan

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_mean = depth_tools._diverging_functions_internal.masked_mean_unchecked(
            a=im, along_all_dims_except_0=False, keep_dropped_dims=True, mask=mask
        )
        expected_mean = im.mean(where=mask, keepdims=True)
        self.assertTrue(np.isnan(expected_mean))  # self test

        self.assertAllclose(actual_mean, expected_mean, equal_nan=True)
        self.assertEqual(len(actual_mean.shape), len(im.shape))

    def test_masked_mean__multiple__no_keepdim(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.stack([im, 2 * im])

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_mean = depth_tools._diverging_functions_internal.masked_mean_unchecked(
            a=im, along_all_dims_except_0=True, keep_dropped_dims=False, mask=mask
        )
        expected_mean = np.stack(
            [
                im[0].mean(where=mask[0]),
                im[1].mean(where=mask[1]),
            ]
        )
        self.assertArrayEqual(
            np.isnan(expected_mean), np.array([False, False])
        )  # self test

        self.assertAllclose(actual_mean, expected_mean)
        self.assertEqual(len(actual_mean.shape), 1)

    def test_masked_mean__multiple__no_keepdim__nan(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.stack([im, 2 * im])
        im[-1, -1, -1] = np.nan

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_mean = depth_tools._diverging_functions_internal.masked_mean_unchecked(
            a=im, along_all_dims_except_0=True, keep_dropped_dims=False, mask=mask
        )
        expected_mean = np.stack(
            [
                im[0].mean(where=mask[0]),
                im[1].mean(where=mask[1]),
            ]
        )
        self.assertArrayEqual(
            np.isnan(expected_mean), np.array([False, True])
        )  # self test

        self.assertAllclose(actual_mean, expected_mean, equal_nan=True)
        self.assertEqual(len(actual_mean.shape), 1)

    def test_masked_mean__multiple__keepdim(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.stack([im, 2 * im])

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_mean = depth_tools._diverging_functions_internal.masked_mean_unchecked(
            a=im, along_all_dims_except_0=True, keep_dropped_dims=True, mask=mask
        )
        expected_mean = np.stack(
            [
                im[0].mean(where=mask[0]),
                im[1].mean(where=mask[1]),
            ]
        )
        expected_mean = np.expand_dims(np.expand_dims(expected_mean, -1), -1)
        self.assertArrayEqual(
            np.isnan(expected_mean).reshape(len(expected_mean)),
            np.array([False, False]),
        )  # self test

        self.assertAllclose(actual_mean, expected_mean, equal_nan=True)
        self.assertEqual(len(actual_mean.shape), len(im.shape))

    def test_masked_mean__multiple__keepdim__nan(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.stack([im, 2 * im])
        im[-1, -1, -1] = np.nan

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_mean = depth_tools._diverging_functions_internal.masked_mean_unchecked(
            a=im, along_all_dims_except_0=True, keep_dropped_dims=True, mask=mask
        )
        expected_mean = np.stack(
            [
                im[0].mean(where=mask[0]),
                im[1].mean(where=mask[1]),
            ]
        )
        self.assertArrayEqual(
            np.isnan(expected_mean), np.array([False, True])
        )  # self test
        expected_mean = np.expand_dims(np.expand_dims(expected_mean, -1), -1)

        self.assertAllclose(actual_mean, expected_mean, equal_nan=True)
        self.assertEqual(len(actual_mean.shape), len(im.shape))

    def test_masked_mean__single__nokeepdim__pt(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.expand_dims(im, 0)
        im_tensor = torch.tensor(im, requires_grad=True)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_mean = (
            depth_tools.pt._diverging_functions_internal.masked_mean_unchecked(
                a=im_tensor,
                along_all_dims_except_0=False,
                keep_dropped_dims=False,
                mask=torch.tensor(mask),
            )
        )
        expected_mean = im.mean(where=mask)
        self.assertFalse(np.isnan(expected_mean))  # self test

        self.assertAllclose(np.array(actual_mean.detach()), expected_mean)
        self.assertGradientsExist(elementwise=False, start=im_tensor, end=actual_mean)
        self.assertEqual(len(actual_mean.shape), 0)

    def test_masked_mean__single__nokeepdim__nan__pt(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.expand_dims(im, 0)
        im[:, -1, -1] = np.nan
        im_tensor = torch.tensor(im, requires_grad=True)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_mean = (
            depth_tools.pt._diverging_functions_internal.masked_mean_unchecked(
                a=im_tensor,
                along_all_dims_except_0=False,
                keep_dropped_dims=False,
                mask=torch.tensor(mask),
            )
        )
        expected_mean = im.mean(where=mask)
        self.assertTrue(np.isnan(expected_mean))  # self test

        self.assertAllclose(
            np.array(actual_mean.detach()), expected_mean, equal_nan=True
        )
        self.assertGradientsExist(elementwise=False, start=im_tensor, end=actual_mean)
        self.assertEqual(len(actual_mean.shape), 0)

    def test_masked_mean__single__keepdim__pt(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.expand_dims(im, 0)
        im_tensor = torch.tensor(im, requires_grad=True)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_mean = (
            depth_tools.pt._diverging_functions_internal.masked_mean_unchecked(
                a=im_tensor,
                along_all_dims_except_0=False,
                keep_dropped_dims=True,
                mask=torch.tensor(mask),
            )
        )
        expected_mean = im.mean(where=mask, keepdims=True)
        self.assertFalse(np.isnan(expected_mean))  # self test

        self.assertAllclose(np.array(actual_mean.detach()), expected_mean)
        self.assertGradientsExist(elementwise=False, start=im_tensor, end=actual_mean)
        self.assertEqual(len(actual_mean.shape), len(im.shape))

    def test_masked_mean__single__keepdim__nan__pt(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.expand_dims(im, 0)
        im[:, -1, -1] = np.nan
        im_tensor = torch.tensor(im, requires_grad=True)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_mean = (
            depth_tools.pt._diverging_functions_internal.masked_mean_unchecked(
                a=im_tensor,
                along_all_dims_except_0=False,
                keep_dropped_dims=True,
                mask=torch.tensor(mask),
            )
        )
        expected_mean = im.mean(where=mask, keepdims=True)
        self.assertTrue(np.isnan(expected_mean))  # self test

        self.assertAllclose(
            np.array(actual_mean.detach()), expected_mean, equal_nan=True
        )
        self.assertGradientsExist(elementwise=False, start=im_tensor, end=actual_mean)
        self.assertEqual(len(actual_mean.shape), len(im.shape))

    def test_masked_mean__multiple__no_keepdim__pt(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.stack([im, 2 * im])
        im_tensor = torch.tensor(im, requires_grad=True)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_mean = (
            depth_tools.pt._diverging_functions_internal.masked_mean_unchecked(
                a=im_tensor,
                along_all_dims_except_0=True,
                keep_dropped_dims=False,
                mask=torch.tensor(mask),
            )
        )
        expected_mean = np.stack(
            [
                im[0].mean(where=mask[0]),
                im[1].mean(where=mask[1]),
            ]
        )
        self.assertArrayEqual(
            np.isnan(expected_mean), np.array([False, False])
        )  # self test

        self.assertAllclose(np.array(actual_mean.detach()), expected_mean)
        self.assertGradientsExist(elementwise=True, start=im_tensor, end=actual_mean)
        self.assertEqual(len(actual_mean.shape), 1)

    def test_masked_mean__multiple__no_keepdim__nan__pt(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.stack([im, 2 * im])
        im[-1, -1, -1] = np.nan
        im_tensor = torch.tensor(im, requires_grad=True)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_mean = (
            depth_tools.pt._diverging_functions_internal.masked_mean_unchecked(
                a=im_tensor,
                along_all_dims_except_0=True,
                keep_dropped_dims=False,
                mask=torch.tensor(mask),
            )
        )
        expected_mean = np.stack(
            [
                im[0].mean(where=mask[0]),
                im[1].mean(where=mask[1]),
            ]
        )
        self.assertArrayEqual(
            np.isnan(expected_mean), np.array([False, True])
        )  # self test

        self.assertAllclose(
            np.array(actual_mean.detach()), expected_mean, equal_nan=True
        )
        self.assertGradientsExist(elementwise=True, start=im_tensor, end=actual_mean)
        self.assertEqual(len(actual_mean.shape), 1)

    def test_masked_mean__multiple__keepdim__pt(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.stack([im, 2 * im])
        im_tensor = torch.tensor(im, requires_grad=True)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_mean = (
            depth_tools.pt._diverging_functions_internal.masked_mean_unchecked(
                a=im_tensor,
                along_all_dims_except_0=True,
                keep_dropped_dims=True,
                mask=torch.tensor(mask),
            )
        )
        expected_mean = np.stack(
            [
                im[0].mean(where=mask[0]),
                im[1].mean(where=mask[1]),
            ]
        )
        self.assertArrayEqual(
            np.isnan(expected_mean), np.array([False, False])
        )  # self test
        expected_mean = np.expand_dims(np.expand_dims(expected_mean, -1), -1)

        self.assertAllclose(np.array(actual_mean.detach()), expected_mean)
        self.assertGradientsExist(elementwise=True, start=im_tensor, end=actual_mean)
        self.assertEqual(len(actual_mean.shape), len(im.shape))

    def test_masked_mean__multiple__keepdim__nan__pt(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.stack([im, 2 * im])
        im[-1, -1, -1] = np.nan
        im_tensor = torch.tensor(im, requires_grad=True)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_mean = (
            depth_tools.pt._diverging_functions_internal.masked_mean_unchecked(
                a=im_tensor,
                along_all_dims_except_0=True,
                keep_dropped_dims=True,
                mask=torch.tensor(mask),
            )
        )
        expected_mean = np.stack(
            [
                im[0].mean(where=mask[0]),
                im[1].mean(where=mask[1]),
            ]
        )
        self.assertArrayEqual(
            np.isnan(expected_mean), np.array([False, True])
        )  # self test
        expected_mean = np.expand_dims(np.expand_dims(expected_mean, -1), -1)

        self.assertAllclose(
            np.array(actual_mean.detach()), expected_mean, equal_nan=True
        )
        self.assertGradientsExist(elementwise=True, start=im_tensor, end=actual_mean)
        self.assertEqual(len(actual_mean.shape), len(im.shape))

    def test_masked_median__single__nokeepdim(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.expand_dims(im, 0)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_median = (
            depth_tools._diverging_functions_internal.masked_median_unchecked(
                a=im, along_all_dims_except_0=False, keep_dropped_dims=False, mask=mask
            )
        )
        expected_median = np.array(np.median(im[mask]))

        self.assertAllclose(actual_median, expected_median)
        self.assertEqual(len(actual_median.shape), 0)

    def test_masked_median__single__nokeepdim__nan(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.expand_dims(im, 0)
        im[:, -1, -1] = np.nan

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_median = (
            depth_tools._diverging_functions_internal.masked_median_unchecked(
                a=im, along_all_dims_except_0=False, keep_dropped_dims=False, mask=mask
            )
        )
        expected_median = np.array([np.nan])

        self.assertAllclose(actual_median, expected_median, equal_nan=True)
        self.assertEqual(len(actual_median.shape), 0)

    def test_masked_median__single__keepdim(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.expand_dims(im, 0)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_median = (
            depth_tools._diverging_functions_internal.masked_median_unchecked(
                a=im, along_all_dims_except_0=False, keep_dropped_dims=True, mask=mask
            )
        )
        expected_median = np.expand_dims(
            np.expand_dims(np.expand_dims(np.median(im[mask]), -1), -1), -1
        )

        self.assertAllclose(actual_median, expected_median)
        self.assertEqual(len(actual_median.shape), len(im.shape))

    def test_masked_median__single__keepdim__nan(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.expand_dims(im, 0)
        im[:, -1, -1] = np.nan

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_median = (
            depth_tools._diverging_functions_internal.masked_median_unchecked(
                a=im, along_all_dims_except_0=False, keep_dropped_dims=True, mask=mask
            )
        )
        expected_median = np.expand_dims(
            np.expand_dims(np.expand_dims(np.median(im[mask]), -1), -1), -1
        )
        self.assertTrue(np.isnan(expected_median))  # self test

        self.assertAllclose(actual_median, expected_median, equal_nan=True)
        self.assertEqual(len(actual_median.shape), len(im.shape))

    def test_masked_median__multiple__no_keepdim(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.stack([im, 2 * im])

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_median = (
            depth_tools._diverging_functions_internal.masked_median_unchecked(
                a=im, along_all_dims_except_0=True, keep_dropped_dims=False, mask=mask
            )
        )
        expected_median = np.stack(
            [
                np.median(im[0][mask[0]]),
                np.median(im[1][mask[1]]),
            ]
        )
        self.assertArrayEqual(
            np.isnan(expected_median), np.array([False, False])
        )  # self test

        self.assertAllclose(actual_median, expected_median)
        self.assertEqual(len(actual_median.shape), 1)

    def test_masked_median__multiple__no_keepdim__nan(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.stack([im, 2 * im])
        im[-1, -1, -1] = np.nan

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_median = (
            depth_tools._diverging_functions_internal.masked_median_unchecked(
                a=im, along_all_dims_except_0=True, keep_dropped_dims=False, mask=mask
            )
        )
        expected_median = np.stack(
            [
                np.median(im[0][mask[0]]),
                np.median(im[1][mask[1]]),
            ]
        )
        self.assertArrayEqual(
            np.isnan(expected_median), np.array([False, True])
        )  # self test

        self.assertAllclose(actual_median, expected_median, equal_nan=True)
        self.assertEqual(len(actual_median.shape), 1)

    def test_masked_median__multiple__keepdim(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.stack([im, 2 * im])

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_median = (
            depth_tools._diverging_functions_internal.masked_median_unchecked(
                a=im, along_all_dims_except_0=True, keep_dropped_dims=True, mask=mask
            )
        )
        expected_median = np.stack(
            [
                np.median(im[0][mask[0]]),
                np.median(im[1][mask[1]]),
            ]
        )
        self.assertArrayEqual(
            np.isnan(expected_median), np.array([False, False])
        )  # self test
        expected_median = np.expand_dims(np.expand_dims(expected_median, -1), -1)

        self.assertAllclose(actual_median, expected_median)
        self.assertEqual(len(actual_median.shape), len(im.shape))

    def test_masked_meidan__multiple__keepdim__nan(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.stack([im, 2 * im])
        im[-1, -1, -1] = np.nan

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_median = (
            depth_tools._diverging_functions_internal.masked_median_unchecked(
                a=im, along_all_dims_except_0=True, keep_dropped_dims=True, mask=mask
            )
        )
        expected_median = np.stack(
            [
                np.median(im[0][mask[0]]),
                np.median(im[1][mask[1]]),
            ]
        )
        self.assertArrayEqual(
            np.isnan(expected_median), np.array([False, True])
        )  # self test
        expected_median = np.expand_dims(np.expand_dims(expected_median, -1), -1)

        self.assertAllclose(actual_median, expected_median, equal_nan=True)
        self.assertEqual(len(actual_median.shape), len(im.shape))

    def test_masked_median__single__nokeepdim__pt(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.expand_dims(im, 0)
        im_tensor = torch.tensor(im, requires_grad=True)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_median = (
            depth_tools.pt._diverging_functions_internal.masked_median_unchecked(
                a=im_tensor,
                along_all_dims_except_0=False,
                keep_dropped_dims=False,
                mask=torch.tensor(mask),
            )
        )
        expected_median = np.array(np.median(im[mask]))
        self.assertFalse(np.isnan(expected_median))  # self test

        self.assertAllclose(np.array(actual_median.detach()), expected_median)
        self.assertGradientsExist(elementwise=False, start=im_tensor, end=actual_median)
        self.assertEqual(len(actual_median.shape), 0)

    def test_masked_median__single__nokeepdim__nan__pt(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.expand_dims(im, 0)
        im[-1, -1, -1] = np.nan
        im_tensor = torch.tensor(im, requires_grad=True)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_median = (
            depth_tools.pt._diverging_functions_internal.masked_median_unchecked(
                a=im_tensor,
                along_all_dims_except_0=False,
                keep_dropped_dims=False,
                mask=torch.tensor(mask),
            )
        )
        expected_median = np.array(np.median(im[mask]))
        self.assertTrue(np.isnan(expected_median))

        self.assertAllclose(
            np.array(actual_median.detach()), expected_median, equal_nan=True
        )
        self.assertGradientsExist(elementwise=False, start=im_tensor, end=actual_median)
        self.assertEqual(len(actual_median.shape), 0)

    def test_masked_median__single__keepdim__pt(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.expand_dims(im, 0)
        im_tensor = torch.tensor(im, requires_grad=True)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_median = (
            depth_tools.pt._diverging_functions_internal.masked_median_unchecked(
                a=im_tensor,
                along_all_dims_except_0=False,
                keep_dropped_dims=True,
                mask=torch.tensor(mask),
            )
        )
        expected_median = np.expand_dims(
            np.expand_dims(np.expand_dims(np.median(im[mask]), -1), -1), -1
        )
        self.assertFalse(np.isnan(expected_median))

        self.assertAllclose(np.array(actual_median.detach()), expected_median)
        self.assertGradientsExist(elementwise=False, start=im_tensor, end=actual_median)
        self.assertEqual(len(actual_median.shape), len(im.shape))

    def test_masked_median__single__keepdim__nan__pt(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.expand_dims(im, 0)
        im[-1, -1, -1] = np.nan
        im_tensor = torch.tensor(im, requires_grad=True)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_median = (
            depth_tools.pt._diverging_functions_internal.masked_median_unchecked(
                a=im_tensor,
                along_all_dims_except_0=False,
                keep_dropped_dims=True,
                mask=torch.tensor(mask),
            )
        )
        expected_median = np.expand_dims(
            np.expand_dims(np.expand_dims(np.median(im[mask]), -1), -1), -1
        )
        self.assertTrue(np.isnan(expected_median))

        self.assertAllclose(
            np.array(actual_median.detach()), expected_median, equal_nan=True
        )
        self.assertGradientsExist(elementwise=False, start=im_tensor, end=actual_median)
        self.assertEqual(len(actual_median.shape), len(im.shape))

    def test_masked_median__multiple__no_keepdim__pt(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.stack([im, 2 * im])
        im_tensor = torch.tensor(im, requires_grad=True)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_median = (
            depth_tools.pt._diverging_functions_internal.masked_median_unchecked(
                a=im_tensor,
                along_all_dims_except_0=True,
                keep_dropped_dims=False,
                mask=torch.tensor(mask),
            )
        )
        expected_median = np.stack(
            [
                np.median(im[0][mask[0]]),
                np.median(im[1][mask[1]]),
            ]
        )
        self.assertArrayEqual(
            np.isnan(expected_median), np.array([False, False])
        )  # self test

        self.assertAllclose(np.array(actual_median.detach()), expected_median)
        self.assertGradientsExist(elementwise=True, start=im_tensor, end=actual_median)
        self.assertEqual(len(actual_median.shape), 1)

    def test_masked_median__multiple__no_keepdim__nan__pt(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.stack([im, 2 * im])
        im[-1, -1, -1] = np.nan
        im_tensor = torch.tensor(im, requires_grad=True)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_median = (
            depth_tools.pt._diverging_functions_internal.masked_median_unchecked(
                a=im_tensor,
                along_all_dims_except_0=True,
                keep_dropped_dims=False,
                mask=torch.tensor(mask),
            )
        )
        expected_median = np.stack(
            [
                np.median(im[0][mask[0]]),
                np.median(im[1][mask[1]]),
            ]
        )
        self.assertArrayEqual(
            np.isnan(expected_median), np.array([False, True])
        )  # self test

        self.assertAllclose(
            np.array(actual_median.detach()), expected_median, equal_nan=True
        )
        self.assertGradientsExist(elementwise=True, start=im_tensor, end=actual_median)
        self.assertEqual(len(actual_median.shape), 1)

    def test_masked_median__multiple__keepdim__pt(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.stack([im, 2 * im])
        im_tensor = torch.tensor(im, requires_grad=True)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_median = (
            depth_tools.pt._diverging_functions_internal.masked_median_unchecked(
                a=im_tensor,
                along_all_dims_except_0=True,
                keep_dropped_dims=True,
                mask=torch.tensor(mask),
            )
        )
        expected_median = np.stack(
            [
                np.median(im[0][mask[0]]),
                np.median(im[1][mask[1]]),
            ]
        )
        self.assertArrayEqual(
            np.isnan(expected_median), np.array([False, False])
        )  # self test
        expected_median = np.expand_dims(np.expand_dims(expected_median, -1), -1)

        self.assertAllclose(np.array(actual_median.detach()), expected_median)
        self.assertGradientsExist(elementwise=True, start=im_tensor, end=actual_median)
        self.assertEqual(len(actual_median.shape), len(im.shape))

    def test_masked_median__multiple__keepdim__nan__pt(self):
        im = np.array(
            [
                [3, 5, 1],
                [0, 4, 2],
                [3, np.nan, 7],
            ],
            dtype=np.float32,
        )
        im = np.stack([im, 2 * im])
        im[-1, -1, -1] = np.nan
        im_tensor = torch.tensor(im, requires_grad=True)

        mask = np.ones_like(im, dtype=np.bool)
        mask[:, 2, 1] = False

        actual_median = (
            depth_tools.pt._diverging_functions_internal.masked_median_unchecked(
                a=im_tensor,
                along_all_dims_except_0=True,
                keep_dropped_dims=True,
                mask=torch.tensor(mask),
            )
        )
        expected_median = np.stack(
            [
                np.median(im[0][mask[0]]),
                np.median(im[1][mask[1]]),
            ]
        )
        self.assertArrayEqual(
            np.isnan(expected_median), np.array([False, True])
        )  # self test
        expected_median = np.expand_dims(np.expand_dims(expected_median, -1), -1)

        self.assertAllclose(
            np.array(actual_median.detach()), expected_median, equal_nan=True
        )
        self.assertGradientsExist(elementwise=True, start=im_tensor, end=actual_median)
        self.assertEqual(len(actual_median.shape), len(im.shape))

    def assertGradientsExist(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        elementwise: bool,
    ):

        if elementwise:
            for end_item in end:
                start.grad = None

                end_item.backward(retain_graph=True)

                start_grad = start.grad
                self.assertIsNotNone(start_grad)
                self.assertAny(abs(np.array(start_grad)) > 0)
        else:
            start.grad = None

            end.mean().backward(retain_graph=True)

            start_grad = start.grad
            self.assertIsNotNone(start_grad)
            self.assertAny(abs(np.array(start_grad)) > 0)
