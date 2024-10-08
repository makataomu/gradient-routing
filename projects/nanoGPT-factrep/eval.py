from contextlib import contextmanager
from typing import List, Tuple

import lm_eval
import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from tqdm import tqdm

from load_model import load_model_for_inference


class NanoLM(LM):
    def __init__(self, nano, encode, decode, device, ablate=None) -> None:
        super().__init__()
        self.nano = nano
        self.encode = encode
        self.decode = decode
        self.device = device
        self.ablate = ablate

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, int]]:
        outputs = []
        print(len(requests))

        for i in tqdm(range(0, len(requests))):
            req = requests[i]

            context, continuation = req.args

            # from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md
            # how this all works (illustrated on a causal decoder-only setup):
            #          CTX      CONT
            # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
            # model  \               \
            # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
            # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

            # print("CONTEXT", context)
            # print("CONTINUATION", continuation)
            context_enc = self.encode(context)
            continuation_enc = self.encode(continuation)

            context_and_continuation_enc = context_enc + continuation_enc

            inp = (
                torch.tensor(context_and_continuation_enc).to(self.device).unsqueeze(0)
            )
            assert len(inp) == 1  # only one batch element at a time
            # roughly based upon https://github.com/EleutherAI/lm-evaluation-harness/blob/543617fef9ba885e87f8db8930fbbff1d4e2ca49/lm_eval/models/huggingface.py#L1117
            with torch.inference_mode():
                # Pass tokenized input to the model
                if self.ablate is None:
                    all_logits, _ = self.nano.forward(
                        inp[:, :-1], None, torch.ones_like(inp[:, :-1])
                    )
                else:
                    all_logits, _ = self.nano.forward_ablated(
                        inp[:, :-1], ablate_idx=self.ablate
                    )
                all_logits_log_likelihood = F.log_softmax(all_logits, dim=-1)
                cont_logits_log_likelihood = all_logits_log_likelihood[
                    :, -len(continuation_enc) :
                ]  # [1, continuation_len, d_vocab]
                greedy_tokens = cont_logits_log_likelihood.argmax(dim=-1)
                cont_toks = (
                    torch.tensor(continuation_enc, dtype=torch.long)
                    .unsqueeze(0)
                    .to(greedy_tokens.device)
                )
                assert cont_logits_log_likelihood.shape[1] == len(
                    continuation_enc
                )  # making sure we are comparing the correct thing
                is_top_answer = (greedy_tokens == cont_toks).all()
                # We need to do gather before we can sum TODO
                cont_logits = torch.gather(
                    cont_logits_log_likelihood, 2, cont_toks.unsqueeze(-1)
                ).squeeze(-1)
                assert cont_logits.shape == (1, len(continuation_enc))
                log_likelihood_for_correct_answer = cont_logits.sum()  # multiplying out conditional probabilities is the same as summing in logspace

            outputs.append(
                (log_likelihood_for_correct_answer.item(), int(is_top_answer.item()))
            )

        return outputs

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float]]:
        raise NotImplementedError  # WMDP eval doesn't use this function, so we don't need to implement it

    def generate_until(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError  # WMDP eval doesn't use this function, so we don't need to implement it

    @contextmanager
    def eval_mode(self):
        current_state = self.nano.training
        self.nano.eval()
        yield
        self.nano.train(current_state)


def eval_nano_tasks(model, device, enc, dec, tasks, ablate):
    nano = NanoLM(model, enc, dec, device, ablate=ablate)
    with nano.eval_mode():
        results = lm_eval.simple_evaluate(
            model=nano,
            tasks=tasks,
        )
    return results["results"]


if __name__ == "__main__":
    exec(open("configurator.py").read())  # overrides from command line or config file
    model, mask_config, enc, dec, ctx, device = load_model_for_inference(
        compile=True, path="best_virology_so_far.pt"
    )
    with ctx:
        ev = eval_nano_tasks(
            model,
            device,
            enc,
            dec,
            [
                # "wmdp_bio_continuation",
                "mmlu_continuation",
                # "mmlu_continuation_college_biology",
                # "mmlu_college_biology",
                # "mmlu_computer_security",
                # "mmlu_college_computer_science",
                # "mmlu_college_biology",
                # "mmlu",
            ],
            ablate=None,
        )
        print(ev)
        eva = eval_nano_tasks(
            model,
            device,
            enc,
            dec,
            [
                # "wmdp_bio_continuation",
                "mmlu_continuation",
                # "mmlu_continuation_college_biology",
                # "mmlu_college_biology",
                # "mmlu_computer_security",
                # "mmlu_college_computer_science",
                # "mmlu_college_biology",
                # "mmlu",
            ],
            ablate=0,
        )
        print(eva)
