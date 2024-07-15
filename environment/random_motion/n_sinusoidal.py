from random import gauss, seed, randint
from numpy import sin, cos


class NSinusoidal:
    """
    Represents a N-Sinusoidal random motion model object.

    Attributes:
        n (int): Number of sinusoidal signals.
        params_amplitude (list): A list of amplitudes for each sinusoidal signal.
        params_pulsation (list): A list of pulsations for each sinusoidal signal.
        params_phase (list): A list of phases for each sinusoidal signal.
        params_offset (list): A list of offsets for each sinusoidal signal.

    Methods:
        __init__(self, s: int = 0, n: int = 4): Initializes a NSinusoidal object.
        sample(self, t): Calculates model movement at time t.
    """

    def __init__(self, s: int = 0, n: int = 4, random: bool = False, max_n: int = 10):
        """
        Initializes a NSinusoidal object.

        Args:
            s (int): Seed value for random number generation. Default is 0.
            n (int): Number of sinusoidal signals. Default is 4.
        """
        if s > 0:
            seed(s)

        # if random is True, assign a random value to n regardless of the input
        if random:
            n = randint(1, max_n)

        # assign n
        self.n = n

        self.params_amplitude = [gauss(0.0, 1) for u in range(n)]
        self.params_pulsation = [gauss(0.01, 1) for u in range(n)]
        self.params_phase = [gauss(0.0, 1) for u in range(n)]
        self.params_offset = [gauss(0.0, 1) for u in range(n)]

    def sample(self, t: float):
        """
        Calculates the model's movement at time t.

        Args:
            t (float): Time value.

        Returns:
            float: The calculated movement of the model at time t.
        """
        x = 0

        for k in range(self.n):
            [a, f, phi, offset] = [
                self.params_amplitude[k],
                self.params_pulsation[k],
                self.params_phase[k],
                self.params_offset[k],
            ]

            if k % 2 == 0:
                x += a * sin(t * (f) + phi) + offset
            else:
                x += a * cos(t * (f) + phi) + offset

        return x
