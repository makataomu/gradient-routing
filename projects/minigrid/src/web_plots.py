import random

from factrep.environments.basic import BasicEnv, BasicEnvConfig, Goal
from factrep.environments.partial_oversight import (
    PartialOversightEnv,
    PartialOversightEnvConfig,
)
from factrep.evaluation import render_action_field
from factrep.models import Agent
from rich.pretty import pretty_repr
from shiny import reactive, req
from shiny.express import input, render, ui

ui.page_opts(title="Agent Action Visualization")

ui.tags.script("""
$(document).on('shiny:inputchanged', function(event) {
  if (event.name != 'changed') {
    Shiny.setInputValue('changed', event.name);
  }
});
""")

ui.input_text(
    "agent_path",
    "Enter the path to the agent model:",
    value="",
    width="100%",
)


@reactive.calc
def load_agent() -> Agent:
    path = input.agent_path()
    loaded_agent = Agent.load(path)
    loaded_agent.eval()
    return loaded_agent


@reactive.calc
def load_env() -> tuple[BasicEnvConfig, BasicEnv]:
    agent = load_agent()

    meta = agent.metadata["env_config"]

    # env_config = BasicEnvConfig(
    #     width=meta.get("width", 11),
    #     height=meta.get("height", 11),
    #     n_terminals_by_kind=meta.get("n_terminals_by_kind", [1, 1]),
    #     randomize_terminal_kinds=meta.get("randomize_terminal_kinds", False),
    #     min_distance=meta.get("min_distance", 0),
    #     has_unique_target=meta.get("has_unique_target", True),
    #     has_target_in_input=meta.get("has_target_in_input", False),
    #     reward_for_target=meta.get("reward_for_target", 1),
    #     reward_for_other=meta.get("reward_for_other", 0.1),
    #     randomize_agent_start=meta.get("randomize_agent_start", False),
    #     pov_observation=meta.get("pov_observation", False),
    #     render_mode="rgb_array",
    #     agent_view_size=meta.get("agent_view_size", 11),
    # )
    #
    env_config = PartialOversightEnvConfig(
        width=meta.get("width", 11),
        height=meta.get("height", 11),
        ratio_oversight=meta.get("ratio_oversight", 0.5),
        rewards=[(1, 1), (1, 1)],
        n_terminals_by_kind=[(1, 1), (1, 1)],
        randomize_terminal_kinds=False,
        min_distance=3,
        has_unique_target=True,
        has_target_in_input=False,
        randomize_agent_start=True,
        pov_observation=False,
        agent_view_size=5,
        render_mode="rgb_array",
    )

    print(pretty_repr(env_config))

    env = PartialOversightEnv.from_config(env_config)
    _ = env.reset(seed=42)
    return env_config, env


with ui.layout_columns(col_widths=(6, 6)):
    with ui.card():
        grid_state = reactive.Value({})
        seen_state = reactive.Value(set())

        @render.express
        def create_grid():
            env_config, _ = load_env()
            with ui.div(
                style=f"display: grid; grid-template-columns: repeat({env_config.width - 2}, 1fr); width: 100%; gap: 1em;"
            ):
                for y in range(1, env_config.height - 1):
                    for x in range(1, env_config.width - 1):
                        with ui.div(
                            style="aspect-ratio: 1; width: 100%; padding: 0; display: grid; grid-template-columns: repeat(2, 1fr); width: 100%; gap: 0em;"
                        ):
                            ui.input_action_button(
                                f"btn_{x}_{y}_3",
                                label="",
                                style="aspect-ratio: 1; width: 100%; padding: 0; opacity: 0.3;",
                            )
                            blue_opacity = (
                                0.3 if grid_state().get((x, y), None) != 0 else 1
                            )
                            ui.input_action_button(
                                f"btn_{x}_{y}_0",
                                label="🟦",
                                style=f"aspect-ratio: 1; width: 100%; padding: 0; opacity: {blue_opacity};",
                            )
                            red_opacity = (
                                0.3 if grid_state().get((x, y), None) != 1 else 1
                            )
                            ui.input_action_button(
                                f"btn_{x}_{y}_1",
                                label="🟥",
                                style=f"aspect-ratio: 1; width: 100%; padding: 0; opacity: {red_opacity};",
                            )
                            agent_opacity = (
                                0.3 if grid_state().get((x, y), None) != 2 else 1
                            )
                            ui.input_action_button(
                                f"btn_{x}_{y}_2",
                                label="🟣",
                                style=f"aspect-ratio: 1; width: 100%; padding: 0; opacity: {agent_opacity};",
                            )

            ui.input_action_button(
                "clear_grid",
                "Clear Grid",
                style="width: 100%; margin-top: 1em;",
            )

            ui.input_action_button(
                "randomize_oversight",
                "Randomize Oversight",
                style="width: 100%; margin-top: 1em;",
            )

        @reactive.Effect
        @reactive.event(input.clear_grid)
        def clear_grid_state():
            grid_state.set({})
            seen_state.set(set())

        @reactive.Effect
        @reactive.event(input.randomize_oversight)
        def randomize_oversight():
            seen_state.set(
                {
                    (x, y)
                    for x in range(1, 10)
                    for y in range(1, 10)
                    if random.random() < 0.25
                }
            )

    @reactive.Effect
    @reactive.event(
        *[
            getattr(input, f"btn_{x}_{y}_{kind}")
            for x in range(1, 11 - 1)
            for y in range(1, 11 - 1)
            for kind in [0, 1, 2, 3]
        ]
    )
    def handle_button_clicks():
        changed = req(input.changed())
        if not changed.startswith("btn_"):
            return
        x, y, kind = changed.split("_")[-3:]
        # Flip the sign of the kind if the button is pressed
        current = grid_state().get((int(x), int(y)), None)
        if current == int(kind):
            if (int(x), int(y)) in seen_state():
                seen_state.set(seen_state() - {(int(x), int(y))})
            else:
                seen_state.set(seen_state() | {(int(x), int(y))})
        else:
            grid_state.set(grid_state() | {(int(x), int(y)): int(kind)})

    with ui.card():

        @render.plot(width=600, height=600)
        def action_field():
            agent = load_agent()
            _, env = load_env()

            terminal_dict = {
                (x, y): Goal(kind=kind)
                for (x, y), kind in grid_state().items()
                if kind in [0, 1]
            }
            env.set_terminals(terminal_dict)
            env.set_seen(seen_state())

            draw_path_from = [
                (x, y) for (x, y), kind in grid_state().items() if kind == 2
            ]

            fig = render_action_field(
                env, agent, "cpu", (5, 5), draw_path_from=draw_path_from
            )
            return fig


with ui.card():
    ui.markdown("#### Agent Config")

    @render.code()
    def agent_config():
        agent = load_agent()
        print(pretty_repr(agent.config, max_width=30))
        return pretty_repr(agent.config, max_width=30)


with ui.card():
    ui.markdown("#### Agent Metadata")

    @render.code()
    def agent_meta():
        agent = load_agent()
        return pretty_repr(agent.metadata, max_width=30)
