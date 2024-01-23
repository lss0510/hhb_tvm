# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument, too-many-lines, import-outside-toplevel
"""calibrate algorithm for quantization."""
import ctypes
import numpy as np
from scipy import stats


from tvm.relay.quantize import _quantize


def _find_scale_by_kl(arr, quantized_dtype="int8", num_bins=8001, num_quantized_bins=255):
    """Given a tensor, find the optimal threshold for quantizing it.
    The reference distribution is `q`, and the candidate distribution is `p`.
    `q` is a truncated version of the original distribution.

    Ref:
    http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    """
    assert isinstance(arr, np.ndarray)
    min_val = np.min(arr)
    max_val = np.max(arr)
    thres = max(abs(min_val), abs(max_val))

    if min_val >= 0 and quantized_dtype in ["uint8"]:
        # We need to move negative bins to positive bins to fit uint8 range.
        num_quantized_bins = num_quantized_bins * 2 + 1

    def get_pointer(arr, ctypes_type):
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes_type))
        return ctypes.cast(ptr, ctypes.c_void_p)

    hist, hist_edges = np.histogram(arr, bins=num_bins, range=(-thres, thres))
    hist_ptr = get_pointer(hist.astype(np.int32), ctypes.c_int)
    hist_edges_ptr = get_pointer(hist_edges, ctypes.c_float)

    max_th = _quantize.FindScaleByKLMinimization(
        hist_ptr, hist_edges_ptr, num_bins, num_quantized_bins
    )
    return -max_th, max_th


def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
    corresponding amount off the non-zero values.
    Ref: http://hanj.cs.illinois.edu/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError("The discrete probability distribution is malformed. All entries are 0.")
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, "n_zeros=%d, n_nonzeros=%d, eps1=%f" % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist


def _find_scale_by_asy_kl(arr, quantized_dtype="uint8", num_bins=2001, num_quantized_bins=255):

    assert isinstance(arr, np.ndarray)

    hist, hist_edges = np.histogram(arr, bins=num_bins)

    thresholds = []
    divergence = []
    quantized_bins = np.zeros(num_quantized_bins, dtype=np.int32)
    for p_bin_idx_start in range(len(hist_edges) - 255):

        p_bin_idx_stop = len(hist_edges) - 1
        sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]
        thresholds.append([hist_edges[p_bin_idx_start], hist_edges[p_bin_idx_stop]])

        p = sliced_nd_hist.copy()

        left_outlier_count = np.sum(hist[0:p_bin_idx_start])
        p[0] += left_outlier_count
        # put right outlier count in p[-1]
        right_outlier_count = np.sum(hist[p_bin_idx_stop:])
        p[-1] += right_outlier_count

        is_nonzeros = (p != 0).astype(np.int32)

        num_merged_bins = sliced_nd_hist.size // num_quantized_bins

        for j in range(num_quantized_bins):
            began = j * num_merged_bins
            stop = began + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[began:stop].sum()

        quantized_bins[-1] += sliced_nd_hist[num_quantized_bins * num_merged_bins :].sum()

        q = np.zeros(sliced_nd_hist.size, dtype=np.float32)
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            if j == num_quantized_bins - 1:
                stop = len(is_nonzeros)
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[p == 0] = 0
        p = _smooth_distribution(p)
        # There is a chance that q is an invalid probability distribution.
        try:
            q = _smooth_distribution(q)
        except ValueError:
            pass
        divergence.append(stats.entropy(p, q))
    min_divergence_idx = np.argmin(divergence)
    opt_th = thresholds[min_divergence_idx]

    return opt_th
