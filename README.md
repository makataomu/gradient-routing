# Gradient routing
A companion repository for [Gradient Routing: Masking Gradients to Localize Computation in Neural Networks](https://arxiv.org/abs/2410.04332).


## Repo structure
* `projects` contains the code to reproduce the results in the paper
    * `minigrid_repro` - localizing behavioral tendencies in a gridworld reinforcement learning agent

Contains experiments with Early Stopping. 
Plots and more information - https://docs.google.com/presentation/d/1UMwnGKdVY3_sytJrz6ZMKd67p-jwfdg6vmkMzbrb17U/edit?usp=sharing

## To use

1. Install [PDM](https://pdm-project.org/)
2. Install the PDM project (ie. install the dependencies)
    ```bash
    pdm install
    ```
3. Install the recommended VSCode extensions
4. Install the pre-commit git hooks
    ```bash
    pdm run pre-commit install
    ```

You can then run Python scripts with `pdm run python <script.py>` or by activating the
virtual environment specified by `pdm info`. Eg:
```bash
source /pdm-venvs/factored-representations-Dp430888-3.12/bin/activate
```

`.vscode/settings.json` is configured to automatically format and lint the code with
[Ruff](https://docs.astral.sh/ruff/) (using the extension) on save.

## Tests

Run the tests with:
```bash
pdm run pytest
```

## Citation
```
@article{cloud2024gradient,
	title={Gradient Routing: Masking Gradients to Localize Computation in Neural Networks},
	url={https://arxiv.org/abs/2410.04332v1},
	journal={arXiv.org},
	author={Cloud, Alex and Goldman-Wetzler, Jacob and Wybitul, Evžen and Miller, Joseph and Turner, Alexander Matt},
	year={2024},
}
```
