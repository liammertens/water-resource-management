#import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPIPDContinuousAction

# Error with the dimensions of the reward array when penalized=False (probably a bug when summing the reward with penalty=0.0)
#   Fixed by downgrading to numpy 1.21 fromn 1.24

env = mo_gym.make('water-reservoir-v0', normalized_action=True, nO=2, penalize=False)
# Bug when penalize=True
# using numpy.random._generator.Generator (default in gym), the np_random.randint method does not work on default np.Generator (dam_env.py line 140)
#   This functionality has been removed in numpy 1.23 (required version in pyproject is 1.21=<)
#   When downgrading to numpy 1.21 bug persists
#   also see gym/utils/seeding.py
#   Perhaps change randint to integers?
#print(env.np_random)

# AttributeError: 'GPIPDContinuousAction' object has no attribute 'dynamics' (line 543 in gpi_pd_continuous_action.py)
#   occurs only when dyna=False (maybe due to if check not having else branch such that self.dynamics=None at line 213?)
#   Fixed: added else branch locally after if-test line 213

# When using the model based version (dyna=True, per=True), NotImplementedError is raised by:
#   morl-baselines/common/model-based/utils.py (env_id==water-reservoir-v0 is not in the if-statement in the contructor of ModelEnv)
agent = GPIPDContinuousAction(env=env, per=True, dyna=False, experiment_name='gpi-ls_2_obj')

agent.train(env, ref_point=np.array([0,0], dtype=np.float32)) #random ref_point

#agent.load("./weights/GPI-PD gpi-ls iter=10.tar")
