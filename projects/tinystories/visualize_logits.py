from typing import List, Optional

import plotly.graph_objects as go
import torch as t
from plotly.subplots import make_subplots
from transformer_lens import HookedTransformer


def plot_word_scores(
    scores: t.Tensor,
    model: HookedTransformer,
    list_len: int = 15,
    title: Optional[str] = None,
    show_bottom: bool = False,
) -> go.Figure:
    words: List[str] = model.to_str_tokens(t.arange(model.cfg.d_vocab).int())  # type: ignore
    n_cols = 3 if show_bottom else 2
    assert scores.size(0) == len(words), "Scores and words must have the same length"

    two_col_spec = [[{"colspan": 2}, None], [{}, {}]]
    three_col_spec = [[{"colspan": 3}, None, None], [{}, {}, {}]]
    two_col_titles = [None, "Median", "Top"]
    three_col_titles = [None, "Bottom", "Median", "Top"]

    fig = make_subplots(
        rows=2,
        cols=n_cols,
        specs=three_col_spec if show_bottom else two_col_spec,
        row_heights=[0.2, 0.8],
        subplot_titles=three_col_titles if show_bottom else two_col_titles,
    )
    # Histogram of scores in the top row
    non_zero_scores = scores[scores != 0]
    fig.add_trace(
        go.Histogram(x=non_zero_scores.cpu().numpy(), nbinsx=100), row=1, col=1
    )
    fig.update_yaxes(tickvals=[], ticktext=[], row=1, col=1)

    # Get the top, middle, and bottom `list_len` words
    top_scores, top_idxs = scores.topk(list_len)

    # Sort the scores and get indices
    sorted_scores, sorted_indices = scores.sort()
    # Calculate the start and end indices for the middle scores
    middle_start = len(scores) // 2 - list_len // 2
    middle_end = middle_start + list_len
    # Extract the middle scores and words
    middle_scores = sorted_scores[middle_start:middle_end]
    middle_idxs = sorted_indices[middle_start:middle_end]

    bottom_scores, bottom_idxs = scores.topk(list_len, largest=False)

    min_score, max_score = scores.min().item(), scores.max().item()

    middle_and_top = [[middle_scores, top_scores], [middle_idxs, top_idxs]]
    bot_middle_top = [
        [bottom_scores, middle_scores, top_scores],
        [bottom_idxs, middle_idxs, top_idxs],
    ]
    idxs_and_scores = bot_middle_top if show_bottom else middle_and_top

    # In the bottom row plot the top, middle, and bottom words as heatmaps
    for i, (s, w) in enumerate(zip(*idxs_and_scores)):
        # Sort s and w by s
        sorted_s, sorted_idxs = s.sort()
        sorted_w = w[sorted_idxs]
        fig.add_trace(
            go.Heatmap(
                z=sorted_s.unsqueeze(-1).cpu().numpy(),
                text=[[words[i]] for i in sorted_w.cpu().numpy()],
                texttemplate="%{text}",
                textfont={"size": 15},
                zmin=min_score,
                zmax=max_score,
            ),
            row=2,
            col=i + 1,
        )
        fig.update_xaxes(tickvals=[], ticktext=[], row=2, col=n_cols - i)
        fig.update_yaxes(tickvals=[], ticktext=[], row=2, col=n_cols - i)
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    fig.update_layout(height=420 + (18 * (list_len - 15)), width=450, title_text=title)
    return fig
