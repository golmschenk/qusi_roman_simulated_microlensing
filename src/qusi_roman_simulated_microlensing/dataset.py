import pandas as pd
from random import shuffle

from functools import partial

from enum import StrEnum

from pathlib import Path
import polars as pl

import numpy as np

from qusi.data import LightCurveDataset, LightCurveObservationCollection
from qusi.internal.light_curve_dataset import default_light_curve_observation_post_injection_transform


class ColumnName(StrEnum):
    TIME = 'time'
    MEASURED_RELATIVE_FLUX = 'measured_relative_flux'
    UNKNOWN2 = 'unknown2'
    TRUE_RELATIVE_FLUX = 'true_relative_flux'


def get_paths():
    paths = list(Path('data/roman_simulated_microlensing').glob('*.det.lc'))
    paths = sorted(paths)
    shuffle(paths)
    return paths


def get_paths_test_validation_train():
    paths = get_paths()
    paths_length = len(paths)
    paths_ten_percent_length = paths_length // 10
    test_paths = paths[:paths_ten_percent_length]
    validation_paths = paths[paths_ten_percent_length:2 * paths_ten_percent_length]
    train_paths = paths[2 * paths_ten_percent_length:]
    return test_paths, validation_paths, train_paths


def get_train_paths() -> list[Path]:
    _, _, train_paths = get_paths_test_validation_train()
    return train_paths


def get_validation_paths() -> list[Path]:
    _, validation_paths, _ = get_paths_test_validation_train()
    return validation_paths


def load_microlensing_times_and_fluxes_path(path):
    data_frame = pl.read_csv(path, comment_prefix='#', columns=[0, 1, 2, 3],
                             new_columns=[element.value for element in ColumnName], separator=' ',
                             schema_overrides={element.value: pl.Float32 for element in ColumnName})
    times = data_frame.get_column(ColumnName.TIME).to_numpy().astype(dtype=np.float32)
    fluxes = data_frame.get_column(ColumnName.MEASURED_RELATIVE_FLUX).to_numpy().astype(dtype=np.float32)
    return times, fluxes


def load_noise_times_and_fluxes_path(path):
    data_frame = pl.read_csv(path, comment_prefix='#', columns=[0, 1, 2, 3],
                             new_columns=[element.value for element in ColumnName], separator=' ',
                             schema_overrides={element.value: pl.Float32 for element in ColumnName})
    times = data_frame.get_column(ColumnName.TIME).to_numpy().astype(dtype=np.float32)
    measured_relative_fluxes = data_frame.get_column(ColumnName.MEASURED_RELATIVE_FLUX).to_numpy().astype(
        dtype=np.float32)
    microlensing_signal_fluxes = data_frame.get_column(ColumnName.TRUE_RELATIVE_FLUX).to_numpy().astype(
        dtype=np.float32)
    fluxes = measured_relative_fluxes / microlensing_signal_fluxes
    return times, fluxes

x = 5

def load_microlensing_times_and_fluxes_path2(path):
    data_frame = pd.read_csv(path, comment='#', delimiter='\s+')
    times = data_frame[0].values.astype(dtype=np.float32)
    fluxes = data_frame[1].values.astype(dtype=np.float32)
    return times, fluxes


def load_noise_times_and_fluxes_path2(path):
    data_frame = pd.read_csv(path, comment='#', delimiter='\s+')
    times = data_frame[0].values.astype(dtype=np.float32)
    measured_relative_fluxes = data_frame[1].values.astype(dtype=np.float32)
    microlensing_signal_fluxes = data_frame[2].values.astype(dtype=np.float32)
    fluxes = measured_relative_fluxes / microlensing_signal_fluxes
    return times, fluxes



def positive_label_function(path):
    return 1


def negative_label_function(path):
    return 0


def get_train_dataset():
    microlensing_light_curve_collection = LightCurveObservationCollection.new(
        get_paths_function=get_train_paths,
        load_times_and_fluxes_from_path_function=load_microlensing_times_and_fluxes_path,
        load_label_from_path_function=positive_label_function)
    noise_light_curve_collection = LightCurveObservationCollection.new(
        get_paths_function=get_train_paths,
        load_times_and_fluxes_from_path_function=load_noise_times_and_fluxes_path,
        load_label_from_path_function=negative_label_function)
    post_injection_transform = partial(
        default_light_curve_observation_post_injection_transform, length=42227
    )
    train_light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=[microlensing_light_curve_collection, noise_light_curve_collection],
        post_injection_transform=post_injection_transform
    )
    return train_light_curve_dataset


def get_validation_dataset():
    microlensing_light_curve_collection = LightCurveObservationCollection.new(
        get_paths_function=get_validation_paths,
        load_times_and_fluxes_from_path_function=load_microlensing_times_and_fluxes_path,
        load_label_from_path_function=positive_label_function)
    noise_light_curve_collection = LightCurveObservationCollection.new(
        get_paths_function=get_validation_paths,
        load_times_and_fluxes_from_path_function=load_noise_times_and_fluxes_path,
        load_label_from_path_function=negative_label_function)
    post_injection_transform = partial(
        default_light_curve_observation_post_injection_transform, length=42227
    )
    train_light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=[microlensing_light_curve_collection, noise_light_curve_collection],
        post_injection_transform=post_injection_transform
    )
    return train_light_curve_dataset


if __name__ == '__main__':
    pass
