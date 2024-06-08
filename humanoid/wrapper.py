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

    cinert_start = 44
    cvel_start = cinert_start + nbody * 10
    qfrc_act_start = cvel_start + nbody * 6
    cfrc_ext_start = qfrc_act_start + dof

    def __init__(self, env: gym.Env):
        super().__init__(env)
        env.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(376 - self.nbody * 15 - self.dof,),
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
        processed_cinert = np.zeros((self.nbody, 3))
        masses = cinert[::10]
        com_x, com_y, com_z = cinert[1::10], cinert[2::10], cinert[3::10]
        inertia_xx = cinert[4::10]
        inertia_yy = cinert[5::10]
        inertia_zz = cinert[6::10]
        inertia_xy = cinert[7::10]
        inertia_xz = cinert[8::10]
        inertia_yz = cinert[9::10]

        for i in range(self.nbody):
            com_norm = np.linalg.norm([com_x[i], com_y[i], com_z[i]])
            inertia_norm = np.linalg.norm(
                [
                    inertia_xx[i],
                    inertia_yy[i],
                    inertia_zz[i],
                    inertia_xy[i],
                    inertia_xz[i],
                    inertia_yz[i],
                ]
            )
            processed_cinert[i, 0] = masses[i]
            processed_cinert[i, 1] = com_norm
            processed_cinert[i, 2] = inertia_norm

        return processed_cinert.flatten()

    def _process_cvel(self, vel: OBS_TYPE) -> OBS_TYPE:
        """
        Process the vel observation to get the magnitude of the linear and angular velocities for each body part.
        This process reduce the size of the observation by nbody * (6 - 2).
        """
        processed_vel = np.zeros((self.nbody, 2))

        for i in range(0, self.nbody):
            linear_vel = vel[i * 6 : i * 6 + 3]
            angular_vel = vel[i * 6 + 3 : i * 6 + 6]
            processed_vel[i, 0] = np.linalg.norm(linear_vel)
            processed_vel[i, 1] = np.linalg.norm(angular_vel)

        return processed_vel.flatten()

    def _process_qfrc_actuator(self, qfrc_actuator: OBS_TYPE) -> OBS_TYPE:
        """
        Process the qfrc_actuator observation to get the magnitude of the actuator forces.
        This process reduce the size of the observation by (dof - 1).
        """
        return np.linalg.norm(qfrc_actuator)

    def _process_cfrc_ext(self, cfrc_ext: OBS_TYPE) -> OBS_TYPE:
        """
        Process the cfrc_ext observation to get the magnitude of the external forces.
        This process reduce the size of the observation by nbody * (6 - 2).
        """
        processed_cfrc_ext = np.zeros((self.nbody, 2))

        for i in range(self.nbody):
            linear_force = cfrc_ext[i * 6 : i * 6 + 3]
            torque = cfrc_ext[i * 6 + 3 : i * 6 + 6]
            processed_cfrc_ext[i, 0] = np.linalg.norm(linear_force)
            processed_cfrc_ext[i, 1] = np.linalg.norm(torque)

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

        return np.concatenate(
            (
                positional_and_velocity_based_values,
                cinert,
                cvel,
                [qfrc_actuator],
                cfrc_ext,
            )
        )
