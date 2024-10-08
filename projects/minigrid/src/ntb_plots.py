# %%
#!%load_ext autoreload
#!%autoreload 2

# %%
import random
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import polars as pl
import torch
from gymnasium.wrappers.record_video import RecordVideo
from lets_plot import *
from moviepy.editor import CompositeVideoClip, TextClip, VideoFileClip, clips_array
from polars import col as c

import projects.minigrid.src.factrep.evaluation as eval
import projects.minigrid.src.ppo_less_modified as ppo
from projects.minigrid.src.factrep import utils
from projects.minigrid.src.factrep.environments.partial_oversight import (
    PartialOversightEnv,
    PartialOversightEnvConfig,
)
from projects.minigrid.src.factrep.models import Agent

LetsPlot.setup_html()
_ = torch.autograd.grad_mode.set_grad_enabled(False)


def transform(clip, label, longest):
    clip = clip.set_end(longest)
    txt_clip = (
        TextClip(label, fontsize=18, color="white", font="Arial-Bold")
        .set_pos("top")  # type: ignore
        .set_duration(longest)
    )
    clip = CompositeVideoClip([clip, txt_clip])

    return clip


def evaluate(agent_path: str, video_dir: str, seed: int = 42):
    agent = Agent.load(agent_path)
    agent.eval()
    env_config = agent.metadata["env_config"]

    config = PartialOversightEnvConfig(
        width=env_config["width"],
        height=env_config["height"],
        ratio_oversight=env_config["ratio_oversight"],
        rewards=env_config["rewards"],
        n_terminals_per_kind=env_config["n_terminals_per_kind"],
        terminal_probabilities=[(1, 0), (1, 0)],
        randomize_terminal_kinds=env_config["randomize_terminal_kinds"],
        min_distance=env_config["min_distance"],
        has_unique_target=env_config["has_unique_target"],
        has_target_in_input=env_config["has_target_in_input"],
        randomize_agent_start=env_config["randomize_agent_start"],
        pov_observation=env_config["pov_observation"],
        agent_view_size=env_config["agent_view_size"],
        require_confirmation=env_config["require_confirmation"],
        render_mode="rgb_array",
    )
    make_env = lambda: PartialOversightEnv.from_config(config)

    env = make_env()
    env.reset(seed=seed)

    video_files = []
    for forced_alpha in [[-1, -1], [1, 0], [0, 1]]:
        envs = gym.vector.SyncVectorEnv(
            [
                lambda: RecordVideo(
                    deepcopy(env),
                    video_folder=video_dir,
                    name_prefix=f"target={forced_alpha}",
                    episode_trigger=lambda x: x == 0,
                )
            ]
        )
        video_files.append(f"{video_dir}/target={forced_alpha}-episode-0.mp4")

        init_obs, _ = envs.reset(seed=seed)

        obs = torch.Tensor(init_obs)
        done = False

        while not done:
            action, *_ = agent.get_action_and_value(
                obs, true_alphas=torch.Tensor([forced_alpha])
            )
            obs, _, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated).any()
            obs = torch.Tensor(obs)

        envs.close()

    clips = [VideoFileClip(file) for file in video_files]
    longest = max(clip.duration for clip in clips)
    clips = [
        transform(clip, label, longest)
        for label, clip in zip(["default", "=> blue", "=> yellow"], clips)
    ]
    stacked_clips = clips_array([clips])
    stacked_clips.write_videofile(f"{video_dir}/combined.mp4")


# %%
device = "cpu"

agent = ppo.Agent.load(
    "/Users/eugen/Downloads/7x7_100p_small_no_gating_newppo_iter=1465.cleanrl_model"
)
env_config = agent.metadata["env_config"]
agent.to(device)
agent.eval()

config = PartialOversightEnvConfig(
    width=env_config["width"],
    height=env_config["height"],
    ratio_oversight=env_config["ratio_oversight"],
    rewards=env_config["rewards"],
    n_terminals_per_kind=env_config["n_terminals_per_kind"],
    terminal_probabilities=[(0, 1), (0, 1)],
    randomize_terminal_kinds=env_config["randomize_terminal_kinds"],
    min_distance=env_config["min_distance"],
    has_unique_target=env_config["has_unique_target"],
    has_target_in_input=env_config["has_target_in_input"],
    randomize_agent_start=env_config["randomize_agent_start"],
    pov_observation=env_config["pov_observation"],
    agent_view_size=env_config["agent_view_size"],
    require_confirmation=env_config["require_confirmation"],
    render_mode="rgb_array",
)
make_env = lambda: PartialOversightEnv.from_config(config)

envs = gym.vector.SyncVectorEnv([make_env])
envs.reset(seed=42)

eval.plot_critic_values(envs, agent, device)

# %%
source_ssrl = (
    pl.read_csv("/Users/eugen/Downloads/filtering/*.csv")
    .group_by("step")
    .agg(discounted_return=pl.col("discounted_return").mean())
    .sort("step")
    .with_columns(pl.lit("filtering").alias("model"))
)

source_routing = (
    pl.read_csv("/Users/eugen/Downloads/wandb_export_2024-10-02T07_35_48.969+01_00.csv")
    .with_columns(
        c("discounted_return").rolling_mean(8).alias("discounted_return"),
        model=pl.lit("routing"),
        oversight_level=pl.lit(0.04),
    )
    .select(
        "oversight_level",
        "discounted_return",
        "step",
        "model",
    )
)

df = (
    pl.concat(
        [
            source_ssrl.with_columns(
                pl.lit(oversight).alias("oversight_level"),
                (c("step") * (1 / oversight)).floor().cast(pl.Int64).alias("step"),
                c("discounted_return").alias("discounted_return"),
            ).select(
                "oversight_level",
                "discounted_return",
                "step",
                "model",
            )
            for oversight in [
                # 0.04,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                1.0,
            ]
        ]
        + [source_routing]
    )
    .sort(["step"])
    .with_columns(
        model=pl.concat_str(
            [
                c("model"),
                pl.lit(" ("),
                (c("oversight_level") * 100).cast(pl.Int64).cast(pl.Utf8),
                pl.lit("%)"),
            ]
        )
    )
    .with_columns(
        model=c("model").cast(
            pl.Enum(
                [
                    "filtering (15%)",
                    "filtering (20%)",
                    "filtering (30%)",
                    "filtering (40%)",
                    "filtering (50%)",
                    "filtering (100%)",
                    "routing (4%)",
                ]
            )
        )
    )
    .sort("model")
)

chart = (
    ggplot(df)
    + geom_line(
        aes(
            x="step",
            y="discounted_return",
            color="model",
            group="oversight_level",
            linetype="model",  # Add linetype for color-blind accessibility
        ),
        alpha=0.7,
    )
    # + scale_x_log10()
    + ggsize(width=600, height=300)
    + ggtitle("Efficiency of Data-filtering Baseline v. Gradient Routing")
    + theme(plot_title=element_text(hjust=0.5))
    + labs(x="Number of total steps in an environment", y="Average return")
    + scale_color_manual(
        values={
            "filtering (15%)": "#19B3FF",
            "filtering (20%)": "#19B3FF",
            "filtering (30%)": "#1999FF",
            "filtering (40%)": "#1980FF",
            "filtering (50%)": "#1966FF",
            "filtering (100%)": "#1933FF",
            "routing (4%)": "#FF1919",
        }
    )
    + scale_linetype_manual(  # Add different line types for each model
        values={
            "filtering (15%)": "solid",
            "filtering (20%)": "dotted",
            "filtering (30%)": "dashed",
            "filtering (40%)": "dotdash",
            "filtering (50%)": "longdash",
            "filtering (100%)": "solid",
            "routing (4%)": "solid",
        }
    )
    + xlim(0, 900_000)
)
chart.show()

ggsave(chart, "routing_vs_ssrl.svg")


# %%
def evaluation_results(agent_path: str, n: int = 500, seed: int = 42):
    utils.seed_everything(seed)

    agent = Agent.load(agent_path)
    env_config = agent.metadata["env_config"]

    config = PartialOversightEnvConfig(
        width=env_config["width"],
        height=env_config["height"],
        ratio_oversight=env_config["ratio_oversight"],
        rewards=env_config["rewards"],
        n_terminals_per_kind=env_config["n_terminals_per_kind"],
        terminal_probabilities=[(0, 1), (0, 1)],
        randomize_terminal_kinds=env_config["randomize_terminal_kinds"],
        min_distance=env_config["min_distance"],
        has_unique_target=env_config["has_unique_target"],
        has_target_in_input=env_config["has_target_in_input"],
        randomize_agent_start=env_config["randomize_agent_start"],
        pov_observation=env_config["pov_observation"],
        agent_view_size=env_config["agent_view_size"],
        require_confirmation=env_config["require_confirmation"],
        render_mode="rgb_array",
    )
    print(config)

    make_env = lambda: PartialOversightEnv.from_config(config)

    envs = gym.vector.SyncVectorEnv([utils.wrap_in_loggers(make_env)])

    envs.reset(seed=seed)

    results = []
    count = 0
    while count < n:
        init_obs, _ = envs.reset()
        result = eval.collect_evaluation_data(
            agent, init_obs, envs, torch.device("cpu"), 2
        )
        # if result[0]["optimum_kind"] == 1 and not result[0]["optimum_seen"]:
        # zones = eval.plot_policy(
        #     envs.envs[0].unwrapped,
        #     agent,
        #     "cpu",
        #     require_confirmation=False,
        #     draw_oversight_frame=False,
        #     num_rollouts=10,
        # )
        # plt.show()
        results += result
        count += 1

    print(env_config["rewards"])

    match env_config["rewards"]:
        case [_, (0, -1)]:
            env_kind = "limit"
        case [_, (1, -1)]:
            env_kind = "punish"
        case _:
            env_kind = "normal"

    data = pl.DataFrame(results).with_columns(
        c("found_kind").cast(pl.Enum(["blue", "red"])),
        oversight_level=pl.lit(env_config["terminal_probabilities"][0][1]),
        kind=pl.lit(env_kind),
    )

    return data


data = []
data.append(
    evaluation_results(
        "...",
        n=100,
        seed=42,
    )
)


# %%
def eval_in_env(
    agent_path: str, env_spec: list[tuple[int, int]], n: int = 500, seed: int = 42
):
    utils.seed_everything(seed)

    agent = ppo.Agent.load(agent_path)
    env_config = agent.metadata["env_config"]

    config = PartialOversightEnvConfig(
        width=env_config["width"],
        height=env_config["height"],
        ratio_oversight=env_config["ratio_oversight"],
        rewards=env_config["rewards"],
        n_terminals_per_kind=env_config["n_terminals_per_kind"],
        terminal_probabilities=env_spec,
        randomize_terminal_kinds=env_config["randomize_terminal_kinds"],
        min_distance=env_config["min_distance"],
        has_unique_target=env_config["has_unique_target"],
        has_target_in_input=env_config["has_target_in_input"],
        randomize_agent_start=env_config["randomize_agent_start"],
        pov_observation=env_config["pov_observation"],
        agent_view_size=env_config["agent_view_size"],
        require_confirmation=env_config["require_confirmation"],
        render_mode="rgb_array",
    )

    make_env = lambda: PartialOversightEnv.from_config(config)

    envs = gym.vector.SyncVectorEnv([make_env])

    envs.reset(seed=seed)

    results = []
    count = 0
    while count < n:
        init_obs, _ = envs.reset()
        result = eval.collect_evaluation_data(
            agent, init_obs, envs, torch.device("cpu"), 0
        )
        results += [{"found_kind": None} | r for r in result]
        count += 1

    data = pl.DataFrame(results).with_columns(
        c("found_kind").cast(pl.Enum(["blue", "red"])),
        oversight_level=pl.lit(env_config["terminal_probabilities"][0][1]),
    )

    return data


data = []
for model in Path("...").glob("*.cleanrl_model"):
    iter = int(model.stem.split("iter=")[1])
    for name, env_spec in [
        ("full_oversight", [(0, 1), (0, 1)]),
        ("no_oversight", [(1, 0), (1, 0)]),
        ("blue_hidden", [(1, 0), (0, 1)]),
        ("red_hidden", [(0, 1), (1, 0)]),
    ]:
        print(name)
        if "6800" in model.stem:
            data.append(
                eval_in_env(model, env_spec, n=400, seed=42).with_columns(
                    spec=pl.lit(name), iter=pl.lit(iter)
                )
            )

# %%
df = pl.concat(
    [
        d.select(
            "found_kind", "steered_to", "num_steps", "oversight_level", "iter", "spec"
        )
        for d in data
        if len(d) > 0
    ]
)
(
    ggplot(df)
    + geom_boxplot(aes(x="iter", y="num_steps", color="found_kind"))
    + facet_wrap("spec", ncol=1)
)

# %%
df.group_by("spec", "iter", "found_kind").agg(c("num_steps").mean(), pl.len()).filter(
    c("iter") == 6800, c("found_kind").is_not_null()
).sort("spec", "found_kind")
# %%

df = (
    pl.concat(
        [
            d.select("found_kind", "steered_to", "oversight_level", "kind")
            for d in data
            if len(d) > 0
        ]
    )
    .filter(c("steered_to") == "0")
    .group_by("oversight_level", "steered_to", "kind")
    .agg(
        good=(c("found_kind") == "blue").mean(),
        bad=(c("found_kind") == "red").mean(),
        none=(c("found_kind").is_null()).mean(),
        count=pl.len(),
    )
    .melt(
        id_vars=["oversight_level", "steered_to", "kind", "count"],
        value_vars=["good", "bad", "none"],
        variable_name="found_color",
        value_name="found_proportion",
    )
    .with_columns(
        std_error=(
            c("found_proportion") * (1 - c("found_proportion")) / c("count")
        ).sqrt()
    )
    .with_columns(
        good_lower=pl.when(c("found_color") == "good")
        .then(c("found_proportion") - 1.96 * c("std_error"))
        .otherwise(None),
        good_upper=pl.when(c("found_color") == "good")
        .then(c("found_proportion") + 1.96 * c("std_error"))
        .otherwise(None),
    )
    .cast({"found_color": pl.Enum(["none", "good", "bad"])})  # Change the order here
)

# Calculate baselines
punish_baseline = df.filter(c("kind") == "punish", c("found_color") == "good")
limit_baseline = df.filter(c("kind") == "limit", c("found_color") == "good")

# Filter the main data to include only "normal" kind
df_normal = df.filter(c("kind") == "normal")

display_df = (
    pl.concat(
        [
            df_normal.select(
                "oversight_level",
                "found_proportion",
                "found_color",
                "good_lower",
                "good_upper",
            ).filter(c("oversight_level").is_between(0.00001, 0.9999)),
        ]
    )
    .sort("oversight_level", "found_color")
    .with_columns(
        oversight_level=pl.when(c("oversight_level") == 0.0)
        .then(0.0001)
        .otherwise(c("oversight_level"))
    )
)

chart = (
    ggplot(
        display_df,
        aes(
            x="oversight_level",
            y="found_proportion",
            fill="found_color",
            group="found_color",
        ),
    )
    + geom_area(position="stack", color="#0F172A", size=1.5, alpha=0.7)
    + geom_errorbar(
        aes(ymin="good_lower", ymax="good_upper"),
        position="stack",
        width=1.5,
        color="#0F172A",
    )
    + scale_fill_manual(
        values=["#EE6666", "#6B7280", "#6CB0D4"], limits=["bad", "none", "good"]
    )
    + geom_point(x=0.04, y=0.7, color="black", size=5, fill="#FFE500", shape=23)
    + scale_y_continuous(format=".0%")
    # + scale_x_continuous(format=".0%")
    + scale_x_log10(
        labels=["0.1%", "1%", "10%", "99.9%"],
        breaks=[0.001, 0.01, 0.1, 0.999],
        expand=[0, 0.7],
    )
    + scale_y_continuous(
        labels=["0%", "25%", "50%", "75%", "100%"],
        breaks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    + labs(
        x="Oversight level during training",
        y="Proportion of test episodes",
        fill="Found color",
    )
    # + ggtitle(
    #     "% of good behavior after steering to good",
    #     subtitle="...when bad is closest, no oversight",
    # )
    + theme(
        # plot_title=element_text(hjust=0.5, size=16),
        # plot_subtitle=element_text(hjust=0.5, size=15),
        axis_text=element_text(angle=0, size=17),
        legend_position="none",
    )
    + ggsize(width=400, height=270)
)
chart

# %%
ggsave(chart, "oversight_level.svg")
# %%

# %%
for seed in range(25):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env()
    evaluate(
        "...",
        video_dir=f"videos/evals_gating/seed-{seed}",
        seed=seed,
    )

print(env.unwrapped.target_color)  # type: ignore

# %%
device = "cpu"

agent = Agent.load(
    "/Users/eugen/Downloads/7x7_100p_no_gating_small_iter=400.cleanrl_model"
)
env_config = agent.metadata["env_config"]
agent.to(device)
agent.eval()

config = PartialOversightEnvConfig(
    width=env_config["width"],
    height=env_config["height"],
    ratio_oversight=env_config["ratio_oversight"],
    rewards=env_config["rewards"],
    n_terminals_per_kind=env_config["n_terminals_per_kind"],
    terminal_probabilities=[(0, 1), (0, 1)],
    randomize_terminal_kinds=env_config["randomize_terminal_kinds"],
    min_distance=env_config["min_distance"],
    has_unique_target=env_config["has_unique_target"],
    has_target_in_input=env_config["has_target_in_input"],
    randomize_agent_start=env_config["randomize_agent_start"],
    pov_observation=env_config["pov_observation"],
    agent_view_size=env_config["agent_view_size"],
    require_confirmation=env_config["require_confirmation"],
    render_mode="rgb_array",
)
make_env = lambda: PartialOversightEnv.from_config(config)


env = make_env()
env.reset(seed=99)

zones = eval.plot_policy(
    env,
    agent,
    "cpu",
    require_confirmation=False,
    draw_oversight_frame=True,
    num_rollouts=50,
)

# %%
# Save the matplotlib plot as PDF
zones.savefig("policy_watershed.svg", format="svg", bbox_inches="tight", dpi=300)

# %%
