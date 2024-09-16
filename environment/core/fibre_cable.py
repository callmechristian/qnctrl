"""
This module contains the FibreLink class which represents a fibre optic cable link.
"""


class FibreLink:
    """
    Represents a fibre optic cable link.

    Attributes:
        cable_length (int): The length of the cable in meters.
        cable_radius (float): The radius of the cable in meters.
        drag_coefficient (float): The drag coefficient for the cable.
    """

    cable_length = 100  # m
    cable_radius = 0.01  # m
    drag_coefficient = (
        0.5  # drag coefficient -- for a cylinder #! will differ for different cables
    )

    def __init__(
        self,
        cable_length: int = 100,
        cable_radius: float = 0.01,
        drag_coefficient: float = 0.5,
    ):
        self.cable_length = cable_length
        self.cable_radius = cable_radius
        self.drag_coefficient = drag_coefficient

    def modify_cable_length(self, new_length: int):
        """
        Modify the length of the cable.

        Args:
            new_length (int): The new length of the cable.
        """
        self.cable_length = new_length

    def modify_cable_radius(self, new_radius: float):
        """
        Modify the radius of the cable.

        Args:
            new_radius (float): The new radius of the cable.
        """
        self.cable_radius = new_radius

    def modify_drag_coefficient(self, new_coefficient: float):
        """
        Modify the drag coefficient of the cable.

        Args:
            new_coefficient (float): The new drag coefficient of the cable.
        """
        self.drag_coefficient = new_coefficient
