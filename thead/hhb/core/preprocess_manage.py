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
"""Manage preprocess utils"""
import logging

import numpy as np

from .common import argument_filter_helper
from .common import AttributeDict
from .common import ALL_ARGUMENTS_INFO
from .common import HHBException
from .common import AttributeDict
from .common import convert_invalid_symbol
from .common import find_index
from .common import parse_mean
from .data_process import DataLayout, convert_data_layout
from .data_process import DataPreprocess


# pylint: disable=invalid-name
logger = logging.getLogger("HHB")


def parse_mean(mean):
    """Parse the mean value .

    Parameters
    ----------
    mean : str or list
        The provided mean value

    Returns
    -------
    mean_list : list[int]
        The mean list
    """
    if isinstance(mean, (list, tuple)):
        return list(mean)
    if "," in mean:
        mean = mean.replace(",", " ")
    mean_list = mean.strip().split(" ")
    mean_list = list([float(n) for n in mean_list if n])
    # if len(mean_list) == 1:
    #     mean_list = mean_list * 3
    return mean_list


def get_dataset_format(data_path):
    """Obtain dataset format

    Parameters
    ----------
    data_path : str
        Dataset path

    Returns
    -------
    data_format : str
        Dataset format, "jpg", "npz" or "mixed"
    """
    data_format = None
    dp = DataPreprocess()
    all_file_path = dp.get_file_list(data_path)

    all_suffix = []
    for input_file in all_file_path:
        suffix = input_file.split(".")[-1]
        all_suffix.append(suffix)
    all_suffix = set(all_suffix)
    if "npz" not in all_suffix:
        data_format = "jpg"
    else:
        if all_suffix - set(["npz"]):
            data_format = "mixed"
        else:
            data_format = "npz"
    return data_format


@argument_filter_helper
def collect_preprocess_config(filtered_args, extra=None):
    """add quantize_config item for hold quantization info"""
    filtered_args.preprocess_config = AttributeDict()
    for k in ALL_ARGUMENTS_INFO["preprocess"]:
        filtered_args.preprocess_config[k] = filtered_args[k]

    filtered_args.preprocess_config.data_mean = parse_mean(
        filtered_args.preprocess_config.data_mean
    )
    filtered_args.preprocess_config.simulate_data_format = (
        get_dataset_format(filtered_args.simulate_data)
        if hasattr(filtered_args, "simulate_data") and filtered_args.simulate_data
        else None
    )
    filtered_args.preprocess_config.calibrate_data_format = (
        get_dataset_format(filtered_args.calibrate_dataset)
        if hasattr(filtered_args, "calibrate_dataset") and filtered_args.calibrate_dataset
        else None
    )
    filtered_args.preprocess_config.data_scale = filtered_args["data_scale"] * (
        1 / filtered_args["data_scale_div"]
    )


@argument_filter_helper
def set_preprocess_params(filtered_args, extra=None):
    """infer preprocess arguments accordding to extra arguments"""
    if not hasattr(filtered_args, "preprocess_config"):
        raise HHBException("Please execute 'collect_preprocess_config' filter first.")
    if not hasattr(extra, "input_shape"):
        raise HHBException("extra has no input_shape attr")
    if len(extra.input_shape) > 1:
        logger.warning(
            "You use only one input shape but this model have %d inputs" % len(extra.input_shape)
        )
    input_shape = extra.input_shape[0]
    # set data layout
    if filtered_args.preprocess_config.data_layout is None:
        logger.info("Infer data layout by input shape...")
        data_layout = DataLayout.get_data_layout(input_shape)
        filtered_args.preprocess_config.data_layout = DataLayout.get_type2str(data_layout)
    else:
        data_layout = DataLayout.get_str2type(filtered_args.preprocess_config.data_layout)

    # if (
    #     filtered_args.preprocess_config.calibrate_data_format is not None
    #     and "npz" in filtered_args.preprocess_config.calibrate_data_format
    # ) or (
    #     filtered_args.preprocess_config.simulate_data_format is not None
    #     and "npz" in filtered_args.preprocess_config.simulate_data_format
    # ):
    #     logger.warning(
    #         "Using npz format dataset as calibrate data or input data, so preprocess will not be executed."
    #     )
    #     filtered_args.preprocess_config.with_preprocess = False
    #     return

    if len(input_shape) != 4:
        return

    gray = False
    if input_shape[DataLayout.get_channel_idx(data_layout)] == 1:
        gray = True
    filtered_args.preprocess_config.gray = gray
    # set data resize
    if data_layout == DataLayout.NCHW:
        data_h_w = (input_shape[2], input_shape[3])
    else:
        data_h_w = (input_shape[1], input_shape[2])
    if not filtered_args.preprocess_config.data_resize:
        filtered_args.preprocess_config.data_resize = data_h_w
    # set target shape
    filtered_args.preprocess_config.target_shape = data_h_w
    # set channel swap: whether needs to convert bgr to rgb
    if filtered_args.preprocess_config.pixel_format == "RGB" and not gray:
        filtered_args.preprocess_config.channel_swap = (2, 1, 0)
    else:
        filtered_args.preprocess_config.channel_swap = (0, 1, 2)
    filtered_args.preprocess_config.with_preprocess = True


class DatasetLoader(object):
    """Load and preprocess dataset in generator way"""

    def __init__(
        self,
        data_path: str,
        preprocess_params: AttributeDict,
        input_shape: list,
        input_name=None,
        batch=1,
        target_layout="NCHW",
    ):
        self.data_path = data_path
        self.pre_params = preprocess_params
        self.input_shape = input_shape
        self.input_name = input_name
        self.batch = batch
        self.target_layout = target_layout

        self._dp = DataPreprocess()
        self.all_file_path = self._dp.get_file_list(self.data_path)

    def get_data(self):
        assert self.batch == 1, "Only support for batch_size=1 while loading npz data."
        # check file type
        if get_dataset_format(self.data_path) == "mixed":
            raise ValueError("Detect mixed .npz file and jpg file, only one of them can be used.")

        for input_file in self.all_file_path:
            suffix = input_file.split(".")[-1]
            if suffix == "npz":
                # self.pre_params.with_preprocess = False
                npz_data = np.load(input_file)
                for i_name in self.input_name:
                    valid_keys = []
                    for k in npz_data.keys():
                        valid_keys.append(convert_invalid_symbol(k))
                    if i_name not in valid_keys:
                        logger.warning(
                            "The input name: {} is not in the passed .npz file input name: {}.".format(
                                i_name, valid_keys
                            )
                        )
                        continue
                split_npz_data = {}
                epoch = list(npz_data.values())[0].shape[0]
                npz_origin_names = [name for name in npz_data]
                npz_valid_names = [convert_invalid_symbol(name) for name in npz_data]
                npz_data_list = [npz_data[name] for name in npz_data]
                for idx, name in enumerate(self.input_name):
                    if name not in npz_valid_names:
                        logger.warning(
                            f"model need input name '{npz_origin_names[find_index(npz_valid_names, name)]}',"
                            "but not find in input data."
                        )
                        continue
                    value = npz_data_list[find_index(npz_valid_names, name)]
                    if list(value.shape) == list(self.input_shape[idx]):
                        split_dataset = [value]
                        epoch = 1
                    else:
                        split_dataset = np.split(value, epoch, axis=0)
                    split_npz_data[name] = split_dataset
                for index in range(epoch):
                    tmp_d = {}
                    for k, v in split_npz_data.items():
                        data_rank = len(v[index].shape)
                        if data_rank == 4:
                            v[index] = convert_data_layout(
                                v[index], self.pre_params.data_layout, self.target_layout
                            )
                        tmp_d[k] = v[index]
                        idx = find_index(self.input_name, k)
                        if idx != -1 and list(self.input_shape[idx]) != list(v[index].shape):
                            if len(list(self.input_shape[idx])) != 0:
                                raise ValueError(
                                    "Input data shape doesn't match required shape: {} vs {}".format(
                                        list(v[index].shape), list(self.input_shape[idx])
                                    )
                                )
                    yield tmp_d
            else:
                if len(self.input_name) > 1:
                    raise ValueError(
                        "The number of model input is more than 1, please " "use .npz file instead."
                    )
                self._dp.load_image(input_file, gray=self.pre_params.gray)
                if len(self._dp.get_data()) != self.batch:
                    continue
                self._dp.img_resize(resize_shape=self.pre_params.data_resize)
                self._dp.img_crop(crop_shape=self.pre_params.target_shape)
                self._dp.channel_swap(self.pre_params.channel_swap)
                if not self.pre_params.add_preprocess_node:
                    self._dp.sub_mean(mean_val=self.pre_params.data_mean)
                    self._dp.data_scale(self.pre_params.data_scale)
                self._dp.data_expand_dim()
                self._dp.data_transpose((0, 3, 1, 2))
                dataset = self._dp.get_data()
                dataset = np.concatenate(dataset, axis=0)
                dataset = convert_data_layout(
                    dataset, self.pre_params.data_layout, self.target_layout
                )
                yield {self.input_name[0]: dataset}
                self._dp.data_clear()


def hhb_preprocess(data_path: str, config, is_generator=False):
    """Convert data provided by data_path into target input data."""
    dl = DatasetLoader(
        data_path,
        config._cmd_config.preprocess_config,
        config._cmd_config.input_shape,
        config._cmd_config.input_name,
    )
    res = dl
    if not is_generator:
        dataset_list = []
        for d in dl.get_data():
            dataset_list.append(d)
        res = dataset_list
    return res
