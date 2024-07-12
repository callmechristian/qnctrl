import pandas as pd
import numpy as np

def load_historical_weather_data(path: str = 'data/open-meteo-43.69N7.19E17m.csv'):
    """
    Load historical weather data from a CSV file. Default file: data/open-meteo-43.69N7.19E17m.csv which is temperature, wind speed and wind direction data from Nice, France.

    Returns:
        pandas.DataFrame: The loaded weather data.
    """
    # Path to the CSV file
    csvfile = path

    # Read the CSV file using pandas
    data = pd.read_csv(csvfile)

    data.dropna(inplace=True)
    data.replace([np.inf, -np.inf], 0, inplace=True)
    data['time'] = data['time'].str.replace('T', ' ')
    data.rename(columns={'temperature_2m (°C)': 'temperature'}, inplace=True)
    data.rename(columns={'wind_speed_10m (km/h)': 'wind_speed'}, inplace=True)
    data.rename(columns={'wind_direction_10m (°)': 'wind_direction'}, inplace=True)
    data.rename(columns={'rain (mm)': 'rain'}, inplace=True)
    
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
    selected_data = data.iloc[k:k+n]
    return selected_data

def select_interpolate_samples(data, n, k, interpolation_type='linear'):
    """
    Select and interpolate n samples starting from index k.

    Args:
        data (pandas.DataFrame): The weather data.
        n (int): The number of samples to interpolate.
        k (int): The starting index.
        interpolation_type (str): The type of interpolation. Default is 'linear'.

    Returns:
        pandas.DataFrame: The interpolated samples.
    """
    selected_data = data.iloc[k:k+n]
    interpolated_data = selected_data.interpolate(method=interpolation_type)
    return interpolated_data
