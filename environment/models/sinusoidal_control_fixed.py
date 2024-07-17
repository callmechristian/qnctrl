"""
This module defines the SimpleControlledFixedEnv class, which simulates a controlled
environment for quantum entanglement propagation with fixed errors.

The SimpleControlledFixedEnv class includes methods to step through the simulation,
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

from ..core import polar_control, entangler, compute_qber
from ..random_motion import NSinusoidal
from ..control import sinusoidal_control

class SinusoidalControlledFixedEnv:
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
        max_t: float = 1440,
        latency: int = 3,
        fixed_error: np.array = np.zeros(12),
        sinusoidal_components: int = 4,
        seed: int = 0,
    ):
        """
        Initializes an instance of SinusoidalControlFixedEnv.

        Parameters:
            t0 (float): The initial time value. Default is 0.
            max_t (float): The maximum simulation time horizon. Default is 0.2.

        Returns:
            None
        """
        # the polarization vector of the pump
        self.H = 1 / np.sqrt(2) * np.matrix([[1], [1]]) # pylint: disable=invalid-name

        self.seed = seed

        self.phi = []
        for i in range(12):
            self.phi.append(NSinusoidal(n=sinusoidal_components, s=seed + i))

        self.t = t0 + 0.0
        """
        The initial time value.

        This variable represents the starting time for the simulation.
        """
        self.max_t = max_t + self.t
        """
        The maximum simulation time horizon.

        This variable represents the maximum simulation time.
        """
        self.delta_t = 1  # speed of the samping
        """
        The speed of the sampling.

        The frequency of the error model: lower values mean less fluctuation.
        """

        self.ctrl_pump = [np.zeros(4) for _ in range(sinusoidal_components)]
        """
        The pump control array.

        This variable represents the control array for the pump. e.g. 
        [np.array([0, 0, 0, 0]) for _ in range(n)] 
        for identity
        """
        self.ctrl_alice = [np.zeros(4) for _ in range(sinusoidal_components)]
        """
        The Alice control array.

        This variable represents the control array for Alice. e.g. 
        [np.array([0, 0, 0, 0]) for _ in range(n)] 
        for identity
        """
        self.ctrl_bob = [np.zeros(4) for _ in range(sinusoidal_components)]
        """
        The Bob control array.

        This variable represents the control array for Bob. e.g. 
        [np.array([0, 0, 0, 0]) for _ in range(n)] 
        for identity
        """
        self.latency = latency
        """
        The control latency.

        This variable represents the number of steps the control is delayed. It also represents 
        the number of steps included in the MDP state.
        """
        self.fixed_error_ctrl_pump = fixed_error[0:4] # type: ignore
        """
        The simulated error (array) for the pump.

        This variable represents the simulated error for the pump. It is used to simulate the error 
        for the pump entanglement propagation.
        """
        self.fixed_error_ctrl_alice = fixed_error[4:8] # type: ignore
        """
        The simulated error (array) for Alice.

        This variable represents the simulated error for Alice. It is used to simulate the error 
        for Alice's entanglement propagation.
        """
        self.fixed_error_ctrl_bob = fixed_error[8:12] # type: ignore
        """
        The simulated error (array) for Bob.

        This variable represents the simulated error for Bob. It is used to simulate the error 
        for Bob's entanglement propagation.
        """
        self.fixed_errors_flags = np.repeat(False, 12)
        """
        The fixed error flags.
        
        This variable represents the flags for whcih errors will be fixed.
        """
        self.ctrl_alice_current = [np.zeros(4) for _ in range(sinusoidal_components)]
        self.ctrl_bob_current = [np.zeros(4) for _ in range(sinusoidal_components)]
        self.ctrl_pump_current = [np.zeros(4) for _ in range(sinusoidal_components)]

        self.qber_history: List[float] = []
        """
        The QBER history.
        
        This variable represents the history of the QBER values as [sample][QBERz, QBERx].
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
        self.setting_inverse = False
        """
        If the control should be applied in inverse.
        
        I.e.: np.linalg(control_array) @ state instead of control_array @ state
        """
        self.setting_single = False
        """
        If the control should be applied in single gate: state @ control_array. This will be done using Alice's gate.
        """
        self.sinusoidal_components = sinusoidal_components
        """
        The number of sinusoidal components.
        """
        
        self.reward_ctr = 0.0

    def step(
        self,
        a_pump: List[np.array] = [np.zeros(4) for _ in range(4)],
        a_alice: List[np.array] =  [np.zeros(4) for _ in range(4)],
        a_bob: List[np.array] = [np.zeros(4) for _ in range(4)],
    ):
        """
        Perform a single step in the environment.

        Args:
            a_pump (List[np.array]): Control input for the pump.
            a_alice (List[np.array]): Control input for Alice.
            a_bob (List[np.array]): Control input for Bob.

        Returns:
            tuple: A tuple containing the current state, reward, and done flag.
        """

        # set self control gates to action
        # print("Actions ctrl:")
        # print(a_pump, a_alice, a_bob)
        # print(f"RECEIVED p: {a_pump} a: {a_alice} b: {a_bob}")
        self.ctrl_pump = sinusoidal_control(self.t, a_pump)
        self.ctrl_alice = sinusoidal_control(self.t, a_alice)
        self.ctrl_bob = sinusoidal_control(self.t, a_bob)
        
        # print(f"p: {self.ctrl_pump} a: {self.ctrl_alice} b: {self.ctrl_bob}")

        # *: assume our MDP state is the size of the latency in control
        for ctrl_latency_counter in range(self.latency + 1):
            # update current time step
            self.t += self.delta_t

            # compute the move the angles based on the motion model or fixed
            phi_move = []
            # concatenate the fixed errors
            _errs = np.concatenate(
                (
                    self.fixed_error_ctrl_pump,
                    self.fixed_error_ctrl_alice,
                    self.fixed_error_ctrl_bob,
                )
            )
            for i in range(12):
                # if the error is fixed, we append the fixed error
                if self.fixed_errors_flags[i]:
                    phi_move.append(_errs[i])
                else:
                    # otherwise we append the random error
                    phi_move.append(self.phi[i].sample(self.t))

            # rotation of the pump in the source -- +
            # *: here is where we do the control with @gate
            pump_polarisation = polar_control(phi_move[0:4]) @ self.H
            if self.setting_single:
                pass
            else:
                if self.setting_inverse:
                    pump_polarisation = (
                        np.linalg.inv(polar_control(self.ctrl_pump_current)) @ pump_polarisation
                    )
                else:
                    # print(f"passed: {self.ctrl_pump_current}")
                    pump_polarisation = polar_control(self.ctrl_pump_current) @ pump_polarisation

            # generation of the entangled state
            entangled_state = entangler(pump_polarisation)
            # rotation of the entangled state during the propagation -- gives
            # entangled state at next time step
            entangled_state_propag = (
                np.kron(polar_control(phi_move[4:8]), polar_control(phi_move[8:12]))
                @ entangled_state
            )

            if self.setting_single:
                entangled_state_propag = polar_control(self.ctrl_alice_current) @ entangled_state_propag
            else:
                # *: here is where we do the control with np.kron
                if self.setting_inverse:
                    entangled_state_propag = (
                        np.kron(
                            np.linalg.inv(polar_control(self.ctrl_alice_current)),
                            np.linalg.inv(polar_control(self.ctrl_bob_current)),
                        )
                        @ entangled_state_propag
                    )
                else:
                    entangled_state_propag = (
                        np.kron(
                            polar_control(self.ctrl_alice_current),
                            polar_control(self.ctrl_bob_current),
                        )
                        @ entangled_state_propag
                    )

            # append the angles for plotting
            self.phi_history.append(phi_move)
            # compute the QBERs
            qbers_current = compute_qber(entangled_state_propag)
            self.qber_history.append(qbers_current)

            self.reward_ctr += self.get_reward()
            reward = 0.0
            # *: update control actual values to the current control values
            if ctrl_latency_counter == self.latency:
                self.ctrl_alice_current = self.ctrl_alice
                self.ctrl_bob_current = self.ctrl_bob
                # print(f"ctrl pump current assigned: {self.ctrl_pump}")
                self.ctrl_pump_current = self.ctrl_pump
                
                # compute reward
                reward = self.reward_ctr
                self.reward_ctr = 0

            # if we exceed max t
            if self.t >= self.max_t:
                self.done = True
                break

        return self.get_states(), reward, self.get_done()

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
        self.done = False
        self.qber_history = []
        self.phi_history = []
        self.ctrl_pump_current = np.zeros(4)
        self.ctrl_alice_current = np.zeros(4)
        self.ctrl_bob_current = np.zeros(4)
        self.ctrl_alice = [np.zeros(4) for _ in range(4)]
        self.ctrl_bob = [np.zeros(4) for _ in range(4)]
        self.ctrl_pump = [np.zeros(4) for _ in range(4)]
        # print(self.ctrl_pump_current)
        s, r, _ = self.step()
        # print(f"reset: {s}")
        return s, r

    def get_qber(self):
        """
        Returns the history of QBER (Quantum Bit Error Rate) as a numpy array.

        Returns:
            numpy.ndarray: The history of QBER values as [QBERz, QBERx].
        """
        return np.array(self.qber_history)

    def get_phi(self):
        """
        Returns the phi history as a numpy array.

        Returns:
            numpy.ndarray: The phi history.
        """
        return np.array(self.phi_history)

    def get_state(self):
        """
        Returns the current state of the environment as a numpy array of the two QBERs.

        Returns:
            np.array(2): first QBERz, then QBERx
        """
        return self.qber_history[-1]
    
    def get_states(self):
        """
        Returns the last states of the environment as a numpy array of the two QBERs of length LATENCY.

        Returns:
            [np.array(2), ...]: first QBERz, then QBERx
        """
        return self.qber_history[-self.latency:]

    def get_reward(self):
        """
        Calculate the reward based on the QBER history.

        Returns:
            float: The reward value.
        """
        qber = self.qber_history[-1]
        # bonus_z = 0
        # if qber[0] < 0.05: #and qber[1] < 0.05:
        #     bonus_z = 0.05
        # bonus_x = 0
        # if qber[1] < 0.05:
        #     bonus_x = 0.05

        # bonus_zx = 0
        # if qber[0] < 0.05 and qber[1] < 0.05:
        #     bonus_zx = 0.1
        
        reward = -qber[0] -qber[1]
        # reward = bonus_zx
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