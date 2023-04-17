#import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPIPDContinuousAction


env = mo_gym.make('water-reservoir-v0', normalized_action=True, nO=2, penalize=True, time_limit=365)

#wrapped_env = mo_gym.MOClipReward(env, 1, -50, 0)

# bug??: algorithm returns nan as values for ccs when normalized_action=False
# C:\Users\liamm\anaconda3\lib\site-packages\mo_gymnasium\envs\water_reservoir\dam_env.py:261: RuntimeWarning: invalid value encountered in multiply penalty = -self.penalize * np.abs(bounded_action - action)
# TEMP Fix: when placing an upper bound on the action other than np.inf, agent trains

# normalized_action=True results in faster learning

# When using the model based version (dyna=True, per=True), NotImplementedError is raised by:
#   morl-baselines/common/model-based/utils.py (env_id==water-reservoir-v0 is not in the if-statement in the contructor of ModelEnv)
GPIAgent = GPIPDContinuousAction(env=env, per=True, dyna=False, experiment_name='gpi-ls_2_obj_NormAction_lessPolicyNoise', policy_noise=0.002, noise_clip=0.005)

GPIAgent.train(365000, env, ref_point=np.array([0,0], dtype=np.float32), timesteps_per_iter=36500, eval_freq=3650) #random ref_point

#agent.load("./weights/GPI-PD gpi-ls iter=10.tar")
