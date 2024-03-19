from random import gauss, seed
from numpy import sin, cos

class ladybug:
    """
    Represents a ladybug random motion model object.

    Attributes:
        params_x (list): A list of 8 parameters used for calculating the ladybug's movement.
    
    Methods:
        __init__(self, s: int = 0): Initializes a ladybug object with optional seed value.
        move(self, t): Calculates ladybug movement at time t.
    """

    def __init__(self, s:int = 0):
        """
        Initializes a ladybug model object.

        Args:
            s (int): Seed value for random number generation. Default is 0.
        """
        if s > 0:
            seed(s)
        self.params_x = [gauss(0., 1.) for u in range(8)]
    
    def move(self, t: float):
        """
        Calculates the ladybug model's movement at time t.

        Args:
            t (float): Time value.

        Returns:
            float: The calculated movement of the ladybug model at time t.
        """
        [ax1, ax2, ax3, ax4, kx1, kx2, kx3, kx4] = self.params_x
        
        x = ax1 * sin(t * (kx1 + 20)) + ax2 * cos(t * (kx2 + 10)) + ax3 * sin(t * (kx3 + 5)) + ax4 * cos(t * (kx4 + 5))
    
        return x