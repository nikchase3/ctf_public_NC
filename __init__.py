# TODO: None.

# Makes note of all environments available.

from gym.envs.registration import register

register (
    id = 'cap-v0',
    entry_point = 'gym_cap\gym_cap\envs\cap_env.py:CapEnv',

)
