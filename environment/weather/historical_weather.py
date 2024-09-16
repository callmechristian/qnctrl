"""
This module provides a function to load historical weather data from a CSV file.
"""
from random import seed, randrange
import pandas as pd # type: ignore

class WeatherModel:
    """
    Represents a weather model object.

    Attributes:
        data (pandas.DataFrame): The weather data.

    Methods:
        load_historical_weather_data(): Load historical weather data from a CSV file.
        select_samples(n, k): Select n samples starting from index k.
        select_interpolate_samples(n, k, interpolation_type='linear'): Select and interpolate n 
        samples starting from index k.
    """

    def __init__(self, s: int = 0):
        """
        Initializes a weather model object.
        """
        if s > 0:
            seed(s)
        self.start_point = randrange(0, 240)
        self.data = self.select_samples(-1, self.start_point)
        self.current_index = self.start_point

    def sample(self):
        """
        Return the next sample from the weather data.

        Returns:
            float: The calculated movement of the ladybug model at time t.
        """
        _ret = self.data[self.current_index]
        self.current_index += 1
        return _ret

    def load_historical_weather_data(self):
        """
        Load historical weather data from a CSV file.

        Returns:
            pandas.DataFrame: The loaded weather data.
        """
        # Path to the CSV file
        csvfile = '../../data/open-meteo-43.69N7.19E17m.csv'

        # Read the CSV file using pandas
        data = pd.read_csv(csvfile)

        data.dropna(inplace=True)
        data['time'] = data['time'].str.replace('T', ' ')
        data.rename(columns={'temperature_2m (°C)': 'temperature'}, inplace=True)
        return data

    def select_samples(self, nr_samples, start_index):
        """
        Select nr_samples samples starting from index k.

        Args:
            nr_samples (int): The number of samples to select.
            start_index (int): The starting index.

        Returns:
            pandas.DataFrame: The selected samples.
        """
        selected_data = self.data.iloc[start_index:start_index+nr_samples]
        return selected_data

    def select_interpolate_samples(self, nr_samples, start_index, interpolation_type='linear'):
        """
        Select and interpolate n samples starting from index k.

        Args:
            nr_samples (int): The number of samples to interpolate.
            start_index (int): The starting index.
            interpolation_type (str): The type of interpolation. Default is 'linear'.

        Returns:
            pandas.DataFrame: The interpolated samples.
        """
        selected_data = self.data.iloc[start_index:start_index+nr_samples]
        interpolated_data = selected_data.interpolate(method=interpolation_type)
        return interpolated_data
