# Training a gridworld agent under partial oversight

This directory covers all reinforcement learning results.

## Directory structure

The environment code lives in `src/factrep/environments` and is based on [MiniGrid](https://github.com/Farama-Foundation/Minigrid). The environment we focus on in the paper is the `PartialOversightEnv` from `src/factrep/environments/partial_oversight.py`.

The most novel part is the modified MoE layer in `src/factrep/layers.py`. The rest of the architecture for the policy is in `src/factrep/models.py` and can be configured by passing an `AgentConfig` object to the constructor.

The training code is in `src/ppo_routed.py` and is based on [CleanRL](https://github.com/vwxyzjn/cleanrl). The code has been heavily modified from the original to support the additional loss term, logging, and the gradient-routed agent. A minimally modified version of PPO that we used for ablation studies with non-routed agents is in `src/ppo_nonrouted.py`.

The plots in the paper were generated using `src/ntb_plots.py` and polished in Figma. Old interactive visualizations of the agent's behavior can be found in `src/web_plots.py`.

## Reproducing results

The main results from the RL section of the paper (fig 5a) can be reproduced by running `src/ppo_routed.py` without changes. To run additional experiments, you can modify the `experiments` variable in the script. For example, to run an experiment with 20% oversight where routing is disabled, you can add an entry to the `experiments` list like this:

```js
{
    "exp_name": "20p_no_routing",
    # dot notation is used to set properties in nested config objects
    "envs.0.terminal_probabilities": [(0.8, 0.2), (0.8, 0.2)],
    "agent.routing_enabled": False,
}
```

The results will be logged to wandb. The models will be saved to the `models` directory, and can be evaluated by running the different cells in `src/ntb_plots.py`.
