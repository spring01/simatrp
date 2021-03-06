
from gym.envs.registration import register


kwargs_t = dict(
    max_rad_len=100,
    step_time=1e2,
    completion_time=1e5,
    min_steps=100,
    termination=True,
    k_prop=1.6e3,
    k_act=0.45,
    k_deact=1.1e7,
    k_ter=1e8,
    mono_init=0.0,
    cu1_init=0.0,
    cu2_init=0.0,
    dorm1_init=0.0,
    mono_unit=0.1,
    cu1_unit=0.004,
    cu2_unit=0.004,
    dorm1_unit=0.008,
    mono_cap=10.0,
    cu1_cap=0.2,
    cu2_cap=0.2,
    dorm1_cap=0.4,
    mono_density=8.73,
    sol_init=0.01,
    sol_cap=0.0,
)
max_episode_steps = 100000

'''
Base environment, with termination
'''
register(
    id='ATRP-pst-v0',
    entry_point='simatrp:ATRPBase',
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_t,
)

'''
Base environment, no termination
'''
kwargs_nt = {**kwargs_t, 'termination': False}
register(
    id='ATRP-psnt-v0',
    entry_point='simatrp:ATRPBase',
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_nt,
)

