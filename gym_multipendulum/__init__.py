from gym.envs.registration import register

register(
    id='multipendulum-v0',
    entry_point='gym_multipendulum.envs:MultipendulumEnv',
)
