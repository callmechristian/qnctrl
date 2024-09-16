"""
This module provides functions to load, process, and interpolate historical weather data.

Functions:
- load_historical_weather_data(path: str, interpolate: bool,
interpolation_values: int) -> pd.DataFrame: Loads historical weather data from
a CSV file.
- select_samples(data: pd.DataFrame, n: int, k: int) -> pd.DataFrame: Selects n
samples starting from index k.
- interpolate_data(data: pd.DataFrame, interpolation_values: int,
interpolation_type: str) -> pd.DataFrame: Interpolates the input dataframe.
"""

import pandas as pd  # type: ignore
import numpy as np
from scipy.interpolate import interp1d  # type: ignore


def load_historical_weather_data(
    path: str = "data/open-meteo-wind-temp-rain.csv",
    interpolate: bool = False,
    interpolation_values: int = 30,
):
    """
    Load historical weather data from a CSV file. Default file:
    data/open-meteo-wind-temp-rain.csv which is temperature, wind speed, rain,
    and wind direction data from Nice, France.

    Args:
        path (str): The path to the CSV file.
        interpolate (bool): Interpolate the data. Default: False.
        interpolation_values (int): The number of values to interpolate between
        each original value. Default: 30.
    Returns:
        pandas.DataFrame: The loaded weather data.
    """
    # Path to the CSV file
    csvfile = path

    # Read the CSV file using pandas
    data = pd.read_csv(csvfile)

    data.dropna(inplace=True)
    data.replace([np.inf, -np.inf], 0, inplace=True)
    data["time"] = data["time"].str.replace("T", " ")
    data.rename(columns={"temperature_2m (°C)": "temperature"}, inplace=True)
    data.rename(columns={"wind_speed_10m (km/h)": "wind_speed"}, inplace=True)
    data.rename(columns={"wind_direction_10m (°)": "wind_direction"}, inplace=True)
    data.rename(columns={"rain (mm)": "rain"}, inplace=True)

    if interpolate:
        data = interpolate_data(
            data, interpolation_values=interpolation_values, interpolation_type="cubic"
        )

    return data


def select_samples(data, n, k):
    """
    Select n samples starting from index k.

    Args:
        data (pandas.DataFrame): The weather data.
        n (int): The number of samples to select.
        k (int): The starting index.

    Returns:
        pandas.DataFrame: The selected samples.
    """
    selected_data = data.iloc[k : k + n]
    return selected_data


def interpolate_data(
    data, interpolation_values: int = 30, interpolation_type: str = "cubic"
):
    """
    Interpolate the input dataframe.

    Args:
        data (pandas.DataFrame): The weather data.
        interpolation_values (int): The number of values to interpolate between
        each original value.
        interpolation_type (str): The type of interpolation to use. Options:
        'cubic' -- TBI: 'linear', 'nearest', 'zero', 'slinear', 'quadratic'

    Returns:
        pandas.DataFrame: The interpolated samples.
    """
    r = [
        [data["temperature"].values],
        [data["rain"].values],
        [data["wind_speed"].values],
        [data["wind_direction"].values],
    ]
    r_new = []

    for _, k in enumerate(r):
        k = k[0]
        # print(k)
        le = len(k)

        # Original x values
        x = np.arange(le)

        # New x values with 30 points between each original value
        x_new = np.linspace(0, le - 1, (le - 1) * interpolation_values + 1)

        # Create cubic interpolation function
        cubic_interp = interp1d(x, k, kind=interpolation_type)

        # Interpolate y values for new x values
        r_new.append([cubic_interp(x_new)])
    interpolated_data = pd.DataFrame(
        {
            "temperature": r_new[0][0],
            "rain": r_new[1][0],
            "wind_speed": r_new[2][0],
            "wind_direction": r_new[3][0],
        }
    )
    return interpolated_data
