#import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPIPDContinuousAction


env = mo_gym.make('water-reservoir-v0', normalized_action=True, nO=2, penalize=True)

wrapped_env = mo_gym.MOClipReward(env, 1, -50, 0)

# bug??: algorithm returns nan as values for ccs when normalized_action=False
# C:\Users\liamm\anaconda3\lib\site-packages\mo_gymnasium\envs\water_reservoir\dam_env.py:261: RuntimeWarning: invalid value encountered in multiply penalty = -self.penalize * np.abs(bounded_action - action)
#

# When using the model based version (dyna=True, per=True), NotImplementedError is raised by:
#   morl-baselines/common/model-based/utils.py (env_id==water-reservoir-v0 is not in the if-statement in the contructor of ModelEnv)
agent = GPIPDContinuousAction(env=env, per=True, dyna=False, experiment_name='gpi-ls_2_obj_pen')

agent.train(100000, env, ref_point=np.array([0,0], dtype=np.float32)) #random ref_point

#agent.load("./weights/GPI-PD gpi-ls iter=10.tar")
