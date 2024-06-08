import numpy as np
import gymnasium as gym
import numpy.typing as npt

from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from typing import Optional
from pathlib import Path
from datetime import datetime
from stable_baselines3 import SAC

SCRIPT_DIR = Path(__file__).resolve().parent
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

    def process_cinert(self, cinert: OBS_TYPE) -> OBS_TYPE:
        """
        Process the cinert observation to get the mass, center of mass, and inertia for each body part.
        This process reduce the size of the observation by nbody * (10 - 3).
        """
        masses = cinert[::10]
        com_x = cinert[1::10]
        com_y = cinert[2::10]
        com_z = cinert[3::10]
        inertia_xx = cinert[4::10]
        inertia_yy = cinert[5::10]
        inertia_zz = cinert[6::10]
        inertia_xy = cinert[7::10]
        inertia_xz = cinert[8::10]
        inertia_yz = cinert[9::10]

        processed_cinert = np.zeros((self.nbody, 3))
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

            # Combine mass, com_norm, and inertia_norm
            processed_cinert[i, 0] = masses[i]
            processed_cinert[i, 1] = com_norm
            processed_cinert[i, 2] = inertia_norm

        return processed_cinert.flatten()

    def process_cvel(self, vel: OBS_TYPE) -> OBS_TYPE:
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

    def process_qfrc_actuator(self, qfrc_actuator: OBS_TYPE) -> OBS_TYPE:
        """
        Process the qfrc_actuator observation to get the magnitude of the actuator forces.
        This process reduce the size of the observation by (dof - 1).
        """
        return np.linalg.norm(qfrc_actuator)

    def process_cfrc_ext(self, cfrc_ext: OBS_TYPE) -> OBS_TYPE:
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
        # Custom observation logic here
        (
            positional_and_velocity_based_values,
            cinert,
            cvel,
            qfrc_actuator,
            cfrc_ext,
        ) = self._extract_observation(observation)

        cinert = self.process_cinert(cinert)
        cvel = self.process_cvel(cvel)
        qfrc_actuator = self.process_qfrc_actuator(qfrc_actuator)
        cfrc_ext = self.process_cfrc_ext(cfrc_ext)

        return np.concatenate(
            (
                positional_and_velocity_based_values,
                cinert,
                cvel,
                [qfrc_actuator],
                cfrc_ext,
            )
        )


def get_humanoid_env() -> gym.Env:
    env = gym.make("Humanoid-v4")
    env = HumanoidCustomObservation(env)
    return env


def load_sac_model(
    env: gym.Env,
    lr: float = 0.00025,
    target_update_interval: int = 4,
    chkpt: Optional[Path] = None,
) -> SAC:
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=lr,
        target_update_interval=target_update_interval,
        verbose=1,
    )

    if chkpt is not None:
        try:
            model.set_parameters(chkpt)
            print(f"Loaded model from {chkpt.name}.")
        except ValueError:
            print(f"Failed to load model from {chkpt.name}.")

    return model


def train(model: SAC, total_timesteps: int, save_dir: Path) -> None:
    interrupted = False
    try:
        model.learn(total_timesteps=total_timesteps, log_interval=10)
    except KeyboardInterrupt:
        interrupted = True

    now = datetime.now()
    if interrupted:
        print("Training interrupted at", now.strftime("%m/%d %H:%M:%S"))

    fname = f"sac_humanoid_{now.strftime('%m%d%H%M')}"
    model.save(save_dir / fname)


def env_test(env: gym.Env) -> None:
    env.reset()
    test_steps = 100
    for _ in range(test_steps):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        if done:
            break
    exit()


def main() -> None:
    env = get_humanoid_env()
    models_dir = SCRIPT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model = load_sac_model(env)
    train(model, 50_000_000, models_dir)


if __name__ == "__main__":
    main()
