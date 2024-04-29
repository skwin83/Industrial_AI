from gymnasium.envs.registration import register

register(
    id="gym_examples_v2/CrowdNav-v0",
    entry_point="gym_examples_v2.envs:CrowdNavEnv",
)
