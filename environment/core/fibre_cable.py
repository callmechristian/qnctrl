class FibreLink:
    cable_length = 100 # m
    cable_radius = 0.01 # m

    drag_coefficient = 0.5 # drag coefficient -- for a cylinder #! will differ for different cables

    def __init__(self, cable_length : int = 100,
                 cable_radius : float = 0.01,
                 drag_coefficient : float = 0.5):
        self.cable_length = cable_length
        self.cable_radius = cable_radius
        self.drag_coefficient = drag_coefficient
