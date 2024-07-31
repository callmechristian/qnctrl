"""
This module defines the VariedControlledFixedEnv class, which simulates a controlled
environment for quantum entanglement propagation with fixed errors.

The VariedControlledFixedEnv class includes methods to step through the simulation,
reset the environment, and retrieve the current state, reward, and other information
about the environment.

The module also includes the following functions:
- step: Perform a single step in the environment.
- reset: Resets the environment to its initial state.
- get_qber: Returns the history of QBER (Quantum Bit Error Rate) as a numpy array.
- get_phi: Returns the phi history as a numpy array.
- get_state: Returns the current state of the environment as a numpy array of the two QBERs.
- get_reward: Calculate the reward based on the QBER history.
- get_done: Check if the current time step is greater than or equal to the maximum time step.
- get_info: Returns the value of the 't' attribute.

This module imports the following packages:
- numpy
- polar_control, entangler, compute_qber from the core module
- LadyBug from the random_motion module
"""

from typing import List
import numpy as np
from scipy.interpolate import interp1d  # type: ignore

from ..core import polar_control, entangler, compute_qber, FibreLink, polarisation_from_force
from ..weather.wind_model import WindModel
from data.utils.data_processing import load_historical_weather_data

class WeatherControlledFixedEnv:
    """
    A class that simulates a controlled environment for quantum entanglement propagation.

    This class simulates a controlled environment for quantum entanglement propagation
    with fixed errors.
    It includes methods to step through the simulation, reset the environment,
    and retrieve the current state, reward, and other information about the environment.

    Attributes:
        H (np.matrix): The polarization vector of the pump.
        phi (list): A list of ladybug instances for error simulation.
        t (float): The current time in the simulation.
        max_t (float): The maximum time for the simulation.
        delta_t (float): The speed of the error fluctuation.
        ctrl_pump (np.array): The control array for the pump.
        ctrl_alice (np.array): The control array for Alice.
        ctrl_bob (np.array): The control array for Bob.
        latency (int): The control latency.
        fixed_error_ctrl_pump (np.array): The simulated error for the pump.
        fixed_error_ctrl_alice (np.array): The simulated error for Alice.
        fixed_error_ctrl_bob (np.array): The simulated error for Bob.
        fixed_errors_flags (np.array): The flags for which errors will be fixed.
        ctrl_alice_current (np.array): The current control array for Alice.
        ctrl_bob_current (np.array): The current control array for Bob.
        ctrl_pump_current (np.array): The current control array for the pump.
        done (bool): A flag indicating whether the simulation is done.
        qber_history (list): A history of the QBER values.
        phi_history (list): A history of the phi values.
    """

    def __init__(
        self,
        t0: float = 0,
        max_steps: int = 400,
        step_time: float = 60,
        latency: int = 3,
        fixed_error: np.array = np.zeros(12),
        fibre_segments: int = 2,
        interpolate_data: bool = False,
        interpolation_values: int = 30,
        output_interp: bool = False,
        output_interp_values: int = 30
    ):
        """
        Initializes an instance of SimpleEnv.

        Parameters:
            t0 (float): The initial time value. Default is 0.
            max_t (float): The maximum simulation time horizon. Default is 0.2.

        Returns:
            None
        """
        # the polarization vector of the pump
        self.H = 1 / np.sqrt(2) * np.matrix([[1], [1]]) # pylint: disable=invalid-name
        
        if interpolate_data:
            self.data = load_historical_weather_data(interpolate=True, interpolation_values=interpolation_values)
        else:
            self.data = load_historical_weather_data()
            
        self.output_interp = output_interp
        """
        The output interpolation flag.
        """
        
        self.output_interp_values = output_interp_values
        """
        The number of interpolation values between each point.
        """

        self.phi = [WindModel(self.data) for _ in range(fibre_segments)]
        
        self.t = t0 + 0.0
        """
        The initial time value.

        This variable represents the starting time for the simulation.
        """
        self.max_t = max_steps * step_time
        """
        The maximum simulation time horizon.

        This variable represents the maximum simulation time.
        """
        self.max_steps = max_steps
        """
        The maximum number of steps.
        """
        self.step_time = step_time
        """
        The equivalent real time for each step.
        """
        self.step_count = 0
        """
        The step count.
        """
        self.total_time : float = 0
        """
        The total time for the simulation, computed from the current step count.
        """
        self.ctrl_pump = np.zeros(4)
        """
        The pump control array.

        This variable represents the control array for the pump. e.g. np.array([0, 0, 0, 0]) 
        for identity
        """
        self.ctrl_alice = np.zeros(4)
        """
        The Alice control array.

        This variable represents the control array for Alice. e.g. np.array([0, 0, 0, 0]) 
        for identity
        """
        self.ctrl_bob = np.zeros(4)
        """
        The Bob control array.

        This variable represents the control array for Bob. e.g. np.array([0, 0, 0, 0]) for identity
        """
        self.latency = latency
        """
        The control latency.

        This variable represents the number of steps the control is delayed. It also represents 
        the number of steps included in the MDP state.
        """
        self.nr_fibre_segments = fibre_segments
        self.fixed_errors = np.zeros(fibre_segments)
        self.fixed_errors_flags = np.repeat(False, fibre_segments)
        """
        The fixed error flags.
        
        This variable represents the flags for whcih errors will be fixed.
        """
        self.ctrl_alice_current = np.zeros(4)
        self.ctrl_bob_current = np.zeros(4)
        self.ctrl_pump_current = np.zeros(4)

        self.qber_history: List[float] = []
        """
        The QBER history.
        
        This variable represents the history of the QBER values.
        """
        self.phi_history: List[np.array]  = []
        """
        The phi history.
        
        This variable represents the history of the phi values.
        """
        self.done = False
        """
        The done flag.
        """
        self.Fs: List[List[float]] = [[] for _ in range(fibre_segments)]
        """
        Stores compute Force applied on fibre segments
        """

    def step(
        self,
        a_pump: np.array = np.zeros(4),
        a_alice: np.array = np.zeros(4),
        a_bob: np.array = np.zeros(4),
    ):
        """
        Perform a single step in the environment.

        Args:
            a_pump (np.array, optional): Control input for the pump. Defaults to np.zeros(4).
            a_alice (np.array, optional): Control input for Alice. Defaults to np.zeros(4).
            a_bob (np.array, optional): Control input for Bob. Defaults to np.zeros(4).

        Returns:
            tuple: A tuple containing the current state, reward, and done flag.
        """

        # set self control gates to action
        self.ctrl_pump = a_pump
        self.ctrl_alice = a_alice
        self.ctrl_bob = a_bob

        # create fibrelink segments
        links = [FibreLink() for _ in range(self.nr_fibre_segments)]

        # *: assume our MDP state is the size of the latency in control
        for ctrl_latency_counter in range(self.latency + 1):
            # update current time step
            self.step_count += 1
            self.total_time = self.step_count * self.step_time

            # compute the move the angles based on the motion model or fixed
            phi_move = []
            for i, link in enumerate(links):
                # if the error is fixed, we append the fixed error
                if self.fixed_errors_flags[i]:
                    phi_move.append(self.fixed_errors[i])
                else:
                    # otherwise we append the random error
                    rot_mat, f = self.phi[i].compute_sample(link)
                    phi_move.append(rot_mat)
                    self.Fs[i].append(f)
           
            # rotation of the pump in the source -- +
            # *: here is where we do the control with @gate
            # ? but this error does not come from the propagation, but from the source generation?
            # pump_polarisation = polar_control(phi_move[0:4]) @ self.H
            pump_polarisation = phi_move[0] @ self.H # ^^^
            pump_polarisation = (
                polar_control(self.ctrl_pump_current) @ pump_polarisation
            )
            # print("State after Pump control")
            # print(pump_polarisation)

            # generation of the entangled state
            entangled_state = entangler(pump_polarisation)
            # rotation of the entangled state during the propagation -- gives
            # entangled state at next time step
            entangled_state_propag = entangled_state
            # print("state after entanglement")
            # print(entangled_state_propag)
            # print("States after phi_move")
            for i, phi in enumerate(phi_move):
                if i < len(phi_move) - 1:
                    kron = np.kron(phi, phi_move[i+1])
                    entangled_state_propag = kron @ entangled_state_propag

            # *: here is where we do the control with np.kron
            # print(np.kron(
            #         polar_control(self.ctrl_alice_current),
            #         polar_control(self.ctrl_bob_current),
            #     ))
            # print(entangled_state_propag)
            entangled_state_propag = (
                np.kron(
                    polar_control(self.ctrl_alice_current),
                    polar_control(self.ctrl_bob_current),
                )
                @ entangled_state_propag
            )

            # *: update control actual values to the current control values
            if ctrl_latency_counter == self.latency:
                self.ctrl_alice_current = self.ctrl_alice
                self.ctrl_bob_current = self.ctrl_bob
                self.ctrl_pump_current = self.ctrl_pump

            # append the angles for plotting
            # self.phi_history.append(phi_move)
            # compute the QBERs
            qbers_current = compute_qber(entangled_state_propag)
            self.qbers_current = qbers_current
            self.qber_history.append(qbers_current)

            # if we exceed max t
            if self.step_count >= self.max_steps:
                self.done = True
                break

        return self.get_state(), self.get_reward(), self.get_done()

    def reset(self):
        """
        Resets the environment to its initial state.

        This method sets the time `t` to 0, marks the environment as not done,
        clears the QBER history and phi history, and calls the `step` method.
        Finally, it returns the current state of the environment.

        Returns:
            state (object): The current state of the environment.
        """
        self.t = 0.0
        self.step_count = 0
        # reset the wind models
        for phi in self.phi:
            phi.reset()
        self.done = False
        self.qber_history = []
        self.phi_history = []
        self.step()
        return self.get_state()

    def get_qber(self):
        """
        Returns the history of QBER (Quantum Bit Error Rate) as a numpy array.

        Returns:
            numpy.ndarray: The history of QBER values.
        """
        if(self.output_interp):
            l = len(self.qber_history)
            z = [self.qber_history[i][0] for i in range(l)]
            x = [self.qber_history[i][1] for i in range(l)]
            
            nl = np.arange(l)

            interpz = interp1d(nl, z, kind="cubic", axis=0)
            interpx = interp1d(nl, x, kind="cubic", axis=0)

            nl_new = np.linspace(0, l - 1, (l - 1) * self.output_interp_values + 1)

            _z = interpz(nl_new)
            _x = interpx(nl_new)
            
            return [np.array(_z), np.array(_x)]
            
        return np.array(self.qber_history)

    # def get_phi(self):
    #     """
    #     Returns the phi history as a numpy array.

    #     Returns:
    #         numpy.ndarray: The phi history.
    #     """
    #     return np.array(self.phi_history)

    def get_state(self):
        """
        Returns the current state of the environment as a numpy array of the two QBERs.

        Returns:
            np.array(2): first QBERz, then QBERx
        """
        return self.qber_history[-1]

    def get_reward(self):
        """
        Calculate the reward based on the QBER history.

        Returns:
            float: The reward value.
        """
        qber = self.qber_history[-1]
        reward = -1 * (qber[0] + qber[1])
        return reward

    def get_done(self):
        """
        Check if the current time step is greater than or equal to the maximum time step.

        Returns:
            bool: True if the current time step is greater than or equal to the maximum
            time step, False otherwise.
        """
        return self.t >= self.max_t

    def get_info(self):
        """
        Returns the value of the 't' attribute.

        Returns:
            The value of the 't' attribute.
        """
        return self.t

    def get_qber_x_current(self):
        """
        Returns the current QBERx value.
        
        Returns:
            The current QBERx value.
        """
        return self.qbers_current[1]
    
    def get_qber_z_current(self):
        """
        Returns the current QBERz value.
        
        Returns:
            The current QBERz value.
        """
        return self.qbers_current[0]
