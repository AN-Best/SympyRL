import torch
import sys
from pathlib import Path
from datetime import datetime
from stable_baselines3 import TD3
import multiprocessing as mp
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter


parent_folder = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_folder / "Env"))

from Acrobot_SRK4_torch import AcrobotEnv_srk4_torch

def make_env():
    def _init():
        env = AcrobotEnv_srk4_torch()
        env = Monitor(env)
        return env
    return _init

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    num_envs = 32
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    
    # TensorBoard log directory (timestamped)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_root = Path("runs") / f"td3_acrobot_{timestamp}"
    tb_root.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_root))

    try:
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=10000,
            batch_size=256,
            device="cuda",
            tensorboard_log=str(tb_root),
        )

        # Provide a name for the SB3 run subfolder inside the tensorboard_log dir
        model.learn(total_timesteps=5000000, tb_log_name="td3_acrobot_run")
        model.save("td3_acrobot")

        print("Training complete! TensorBoard logs saved to:", str(tb_root))

    finally:
        writer.close()
        env.close()