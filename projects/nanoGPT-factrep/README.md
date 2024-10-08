# nanoGPT-factrep

Modified from Karpathy's nanoGPT implementation. Thanks Karpathy!

## How to train Steering Scalar Model

Run `data/fineweb-edu/prepare_gpt2.py` to generate `gpt2_train.bin` and `gpt2_val.bin`. These are data / validation files that `california-steering-scalar.py` uses.

Then run `torchrun --standalone --nproc_per_node={how many gpus you want to train on} california-steering-scalar.py` to train the model and you can find the checkpoints in `out`.

Then run `python sample.py` to get the tokens that have highest cosine similarity to the 0th residual stream dimension and print out some steered and unsteered completions. You may have to change `steer_with` which is the coefficient of the steering vector since sometimes the model learns the California feature in the `[1, 0, 0, 0, ...]` direction but other times it learns it in the `[-1, 0, 0, 0, ...]` direction.

## How to train the Virology Unlearned Model

First download the WMDP corpus and put it in the project root with a structure like this:
```
.wmdp/wmdp-corpora/bio-retain-corpus.jsonl
.wmdp/wmdp-corpora/bio-forget-corpus.jsonl
.wmdp/wmdp-corpora/cyber-forget-corpus.jsonl
.wmdp/wmdp-corpora/cyber-retain-corpus.jsonl
```

Then run `data/rineweb-edu/prepare_forget.py` to tokenize the WMDP-bio forget dataset into into into `wmdp_to_train_on.bin` and `wmdp_to_eval_on.bin`. These are the files that are used in the retraining evals for retraining and evaluation respectively. The `wmdp_to_train_on.bin` only contains 2 examples while `wmdp_to_eval_on.bin` contains many more to get an accurate evaluation.

Then run `data/fineweb-edu/prepare.py` so that it will generate `train.bin` and `val.bin`. These files contain the 10 billion token split of the FineWeb-Edu dataset and `train.bin` is used in coherence finetuning.
Finally, modify `data/fineweb-edu/prepare.py` so that it uses the 100BT split of FineWeb-Edu by changing `sample-10BT` to `sample-100BT` and change `sprinkle_wmdp_in_nx` to `5`. This will take a lot of time to download and tokenize the data. What changing `sprinkle_wmdp_in_nx` to `5` does is to shuffle in five of the WMDP-bio forget dataset in the pretraining data to make sure that the model has seen virology and knows how to predict it.

To train the model, you can run `torchrun --standalone --nproc_per_node={num gpus to train on} train.py` to train the model and it should log the evals that you want to W&B. Alternatively, you can use `coherence_ft_and_retrain_evals.py` to test it. We stopped training after `10k` iterations due to time constraints.
