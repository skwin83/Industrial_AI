from gymnasium.envs.registration import register

register(
    id="gym_examples/CrowdNav-v0",
    entry_point="gym_examples.envs:CrowdNavEnv",
)
