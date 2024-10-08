# type: ignore
# %%
import json
import os
from typing import Callable

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from datasets import load_dataset
from nltk.corpus import stopwords
from tqdm import tqdm

from factored_representations import string_utils

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

STOP_WORDS = set(stopwords.words("english"))


def get_parts_of_speech(story: str):
    words = nltk.word_tokenize(story)
    words_with_pos = nltk.pos_tag(words)
    filtered = [
        (word.lower(), pos)
        for word, pos in words_with_pos
        if word not in STOP_WORDS and word.isalnum()
    ]
    df_pos = pd.DataFrame(filtered, columns=["word", "pos"])
    word_pos_freq = df_pos.value_counts().sort_values(ascending=False).reset_index()
    return word_pos_freq


def has_word_pos(story: str, word: str, pos_condition: Callable):
    words_with_pos = nltk.pos_tag(nltk.word_tokenize(story))
    for story_word, pos in words_with_pos:
        if story_word.lower() == word.lower() and pos_condition(pos):
            return True
    return False


broad_parts_of_speech = {
    "improper_nouns": ["NN"],
    "proper_nouns": ["NNP"],
    "verbs": ["VBD", "VB", "VBN", "VBZ", "VBG", "VBP"],
    "adjectives": ["JJ", "JJS", "JJR"],
}

if __name__ == "__main__":
    dataset = load_dataset("delphi-suite/stories")["train"]  # type: ignore
    story = dataset[0]["story"]

    num_stories = 10000
    np.random.seed(42)
    random_indexes = np.random.choice(
        len(dataset), size=num_stories, replace=False
    ).tolist()
    stories = [dataset[idx]["story"] for idx in tqdm(random_indexes)]
    all_stories = " ".join(stories)

    snippets = string_utils.target_word_snippets(all_stories, [" forest", " woodland"])

    PRODUCE_WORD_COUNTS = False
    if PRODUCE_WORD_COUNTS:
        OVERWRITE_FILE = False
        top_words_path = "top_words.json"
        if os.path.exists(top_words_path) and not OVERWRITE_FILE:
            with open(top_words_path, "r") as json_file:
                top_by_pos = json.load(json_file)
        else:
            all_stories = " ".join(stories)
            dfp = get_parts_of_speech(all_stories)

            top_k_words = 100
            top_by_pos = {}
            for label, pos_list in broad_parts_of_speech.items():
                pos_words = dfp[dfp.pos.isin(pos_list)].copy()
                pos_words["pct"] = pos_words["count"] / num_stories
                top = pos_words[:top_k_words]
                top_by_pos[label] = {word: pct for word, pct in zip(top.word, top.pct)}

            with open(top_words_path, "w") as json_file:
                json.dump(top_by_pos, json_file, indent=4)

        dfs = []
        for pos_broad, pos_list in top_by_pos.items():
            df = pd.DataFrame(
                pos_list.items(), columns=["word", "frequency"]
            ).sort_values("frequency")
            df["pos"] = pos_broad
            df[df.pos == pos_broad][-30:].plot(
                "word",
                "frequency",
                kind="barh",
                title=f"Avg. number of each of {pos_broad} per story",
            )

        num_stories = 50
        num_words = 75

        stories_sub = stories[:num_stories]

        dfs = {}
        for pos_broad in broad_parts_of_speech.keys():
            pos_condition = lambda pos: pos in broad_parts_of_speech[pos_broad]
            has_word = {
                word: [
                    has_word_pos(story, word, pos_condition) for story in stories_sub
                ]
                for word in tqdm(list(top_by_pos[pos_broad].keys())[:num_words])
            }
            df = pd.DataFrame(
                {"stories": stories_sub, **has_word}, index=random_indexes[:num_stories]
            )
            dfs[pos_broad] = df

        for pos_broad, df in dfs.items():
            fig, ax = plt.subplots(figsize=(20, 20))
            ax.imshow(df.iloc[:num_stories, 1 : num_words + 1].values.T)
            ax.set_yticks(range(len(df.columns) - 1))
            ax.set_yticklabels(list(df.columns[1:]), fontsize=12)
            ax.set_xlabel("Story")
            ax.set_title(f"{pos_broad} in {num_stories} sampled stories")
