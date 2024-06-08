import numpy as np
import gymnasium as gym
import numpy.typing as npt

from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

OBS_TYPE = npt.NDArray[np.float64]


class HumanoidCustomObservation(ObservationWrapper):
    """Custom observation wrapper for the Humanoid environment."""

    nbody = 14
    dof = 23

    cinert_start = 45
    cvel_start = cinert_start + nbody * 10
    qfrc_act_start = cvel_start + nbody * 6
    cfrc_ext_start = qfrc_act_start + dof

    obs_dim = (376 - nbody * 15 - (dof - 1),)

    def __init__(self, env: gym.Env):
        super().__init__(env)
        env.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=self.obs_dim,
            dtype=np.float64,
        )

    def _extract_observation(
        self, observation: OBS_TYPE
    ) -> tuple[OBS_TYPE, OBS_TYPE, OBS_TYPE, OBS_TYPE, OBS_TYPE]:
        positional_and_velocity_based_values = observation[: self.cinert_start]
        cinert = observation[self.cinert_start : self.cvel_start]
        cvel = observation[self.cvel_start : self.qfrc_act_start]
        qfrc_actuator = observation[self.qfrc_act_start : self.cfrc_ext_start]
        cfrc_ext = observation[self.cfrc_ext_start : 376]

        return (
            positional_and_velocity_based_values,
            cinert,
            cvel,
            qfrc_actuator,
            cfrc_ext,
        )

    def _process_cinert(self, cinert: OBS_TYPE) -> OBS_TYPE:
        """
        Process the cinert observation to get the mass, center of mass, and inertia for each body part.
        This process reduce the size of the observation by nbody * (10 - 3).
        """
        cinert = cinert.reshape(-1, 10)
        masses = cinert[:, 0]
        com = cinert[:, 1:4]
        inertia = cinert[:, 4:]

        com_norm = np.linalg.norm(com, axis=1)
        inertia_norm = np.linalg.norm(inertia, axis=1)

        processed_cinert = np.column_stack((masses, com_norm, inertia_norm))

        return processed_cinert.flatten()

    def _process_cvel(self, cvel: OBS_TYPE) -> OBS_TYPE:
        """
        Process the cvel observation to get the magnitude of the linear and angular velocities for each body part.
        This process reduce the size of the observation by nbody * (6 - 2).
        """
        cvel = cvel.reshape(-1, 6)
        linear_vel = cvel[:, :3]
        angular_vel = cvel[:, 3:]
        linear_vel_norm = np.linalg.norm(linear_vel, axis=1)
        angular_vel_norm = np.linalg.norm(angular_vel, axis=1)
        processed_cvel = np.column_stack((linear_vel_norm, angular_vel_norm))
        return processed_cvel.flatten()

    def _process_qfrc_actuator(self, qfrc_actuator: OBS_TYPE) -> OBS_TYPE:
        """
        Process the qfrc_actuator observation to get the magnitude of the actuator forces.
        This process reduce the size of the observation by (dof - 1).
        """
        return np.array([np.linalg.norm(qfrc_actuator)])

    def _process_cfrc_ext(self, cfrc_ext: OBS_TYPE) -> OBS_TYPE:
        """
        Process the cfrc_ext observation to get the magnitude of the external forces.
        This process reduce the size of the observation by nbody * (6 - 2).
        """
        cfrc_ext = cfrc_ext.reshape(-1, 6)
        linear_force = cfrc_ext[:, :3]
        torque = cfrc_ext[:, 3:]
        linear_force_norm = np.linalg.norm(linear_force, axis=1)
        torque_norm = np.linalg.norm(torque, axis=1)
        processed_cfrc_ext = np.column_stack((linear_force_norm, torque_norm))
        return processed_cfrc_ext.flatten()

    def observation(self, observation: OBS_TYPE) -> OBS_TYPE:
        """
        Process the observation to get the positional and velocity based values, cinert, cvel, qfrc_actuator, and cfrc_ext.
        """
        (
            positional_and_velocity_based_values,
            cinert,
            cvel,
            qfrc_actuator,
            cfrc_ext,
        ) = self._extract_observation(observation)

        cinert = self._process_cinert(cinert)
        cvel = self._process_cvel(cvel)
        qfrc_actuator = self._process_qfrc_actuator(qfrc_actuator)
        cfrc_ext = self._process_cfrc_ext(cfrc_ext)

        transformed_obs = np.concatenate(
            (
                positional_and_velocity_based_values,
                cinert,
                cvel,
                qfrc_actuator,
                cfrc_ext,
            )
        )

        return transformed_obs
