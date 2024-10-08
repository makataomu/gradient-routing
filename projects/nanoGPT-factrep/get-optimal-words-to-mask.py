# %%
from collections import Counter

from transformers import AutoTokenizer

from projects.wmdp import wmdp_loader

# %%

qwen2_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

# %%
wiki_virology_and_children_tokenized = qwen2_tokenizer.encode(
    open("full_text_of_virology_and_children.txt").read()
)

# %%
wiki_virology_and_children_cnt = Counter(wiki_virology_and_children_tokenized)

# %%
wmdp_data = wmdp_loader.load_wmdp_data(section="bio", min_len=1000)
wmdp_forget_data = wmdp_data["forget"][
    :2000
]  # 2000 here so we don't spend too long tokenizing
wmdp_forget_data_concatted = " ".join(wmdp_forget_data)
forget_data_tokenized = qwen2_tokenizer.encode(wmdp_forget_data_concatted)
forget_data_cnt = Counter(forget_data_tokenized)
# %%
wmdp_retain_data = wmdp_data["retain"][:2000]  # this is actually just wikitext
wmdp_retain_data_concatted = " ".join(wmdp_retain_data)
retain_data_tokenized = qwen2_tokenizer.encode(wmdp_retain_data_concatted)
retain_data_cnt = Counter(retain_data_tokenized)


# %%
# normalize each of the counts
def normalize_counter(cnt):
    total = sum(cnt.values())
    return {k: v / total for k, v in cnt.items()}


wiki_normalized = normalize_counter(wiki_virology_and_children_cnt)
forget_normalized = normalize_counter(forget_data_cnt)
retain_normalized = normalize_counter(retain_data_cnt)

# for token in vocab:
#   token_score = min(freq(token, forget_set), freq(token, wiki_bio_and_virology)) / freq(token, wikipedia)
# implement this pseudocode. use forget normalized for the tokens
token_scores = {
    token: min(forget_normalized[token], wiki_normalized.get(token, 0))
    / retain_normalized.get(token, 0.1)
    for token in forget_normalized
}
k = 100
# get the k tokens with the highest score
optimal_tokens = sorted(token_scores, key=token_scores.get, reverse=True)[:k]
optimal_words = [
    (qwen2_tokenizer.decode([token]), token_scores[token]) for token in optimal_tokens
]
print(optimal_words)
