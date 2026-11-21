# Architecture track

Change anything about the model (the weights, the architecture code and the tokenizer) and compete on **validation loss** under a fixed compute budget. The starting point is **Qwen3.5 0.8B Base** (`Qwen/Qwen3.5-0.8B-Base`, the pretrained checkpoint before post-training), packaged as a bundle.

There are two templates:

- **`vllm`** (`delta_cli/arch/baseline/delta_model.py`) is the official baseline: the released weights served by vLLM inside the sandbox. Its speed sets the time limit.
- **`torch`** (`delta_model_hf.py`) is the same model in plain PyTorch/transformers. It is the place to start changing the architecture. It is slower than vLLM, so budget for optimizing your own implementation.

## Score

- **Leaderboard score:** **bits per byte** (lower is better) on a private held-out set of web documents. Bits per byte is tokenizer-free: the total negative log-likelihood of the text divided by its UTF-8 size, so any tokenizer competes fairly.
- **Validation score:** bits per byte on the leading documents of the public FineWeb window used by the modded-nanogpt speedrun. `loss_per_gpt2_token` on that window is also reported, for comparison with the speedrun scale.
- **Document framing:** each document is scored independently. Your model predicts every token of the document and then your `eos_id`. A document longer than `max_context` is split into windows, and each window restarts from `bos_id`.

## Bundle contract

A bundle is a flat directory of regular files. It may contain `.py`, `.safetensors`, `.json`, `.txt`, `.model`, `.tiktoken`, `.jinja` and `.md` files: at most 256 files, 1 MB of Python and the model's byte cap overall. Pickle formats are not accepted. The entry point is `delta_model.py`:

```python
class Model:
    vocab_size: int      # ids are in [0, vocab_size)
    bos_id: int          # input token that starts every window
    eos_id: int          # predicted after the last token of each document
    max_context: int     # longest window you score
    batch_tokens: int    # optional: padded tokens per token_logprobs call

    def __init__(self, bundle_dir: str, device: str): ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def token_logprobs(self, ids, targets):   # LongTensor[B,T] x2 -> FloatTensor[B,T]
        """log P(targets[b,t] | ids[b,:t+1]). Batches are right-padded with bos_id."""
    def score_windows(self, inputs, targets): # optional; list[list[int]] x2 -> list[list[float]]
        """Unpadded alternative to token_logprobs, for engines that batch internally."""
    def next_logits(self, ids):               # LongTensor[1,T] -> FloatTensor[1,V]
        """Unnormalized next-token logits after the last input position."""
```

`delta_cli.arch.helpers.gather_logprobs` computes `token_logprobs` from hidden states without materializing full logits.

### Exact audits

The audit compares numbers computed for the same position in two different shapes: inside a long batched window during scoring, and as a prefix alone during the audit. Make your implementation give the same result regardless of batch composition and sequence length. Deterministic, batch-invariant kernels help. The vLLM baseline is deterministic and batch-invariant in practice, but its numerics depend on sequence length (audit drift std about 0.05 nats; vLLM's batch-invariant mode does not support Qwen3.5's linear-attention layers). Small differences like these are tolerated, and any positive average is charged back to your score, so non-invariant numerics can only cost you.

## How scoring keeps everyone honest

Your code runs in a container with no network, a read-only file system and an empty `/tmp`. Everything it returns is a **claim**, and the harness verifies the claims:

1. **Phase A (timed):** encodes every document and claims a log-probability for every target token.
2. **Phase B (a fresh container):** runs only after phase A has committed its claims.
   - **Audit:** randomly chosen positions (1,000 by default) are re-scored. The model sees only the prefix and returns a full logit row; the harness applies `log_softmax` itself and compares the result with your claim.
   - **Ordering:** for each document, the runner always answers a shorter prefix before it is shown a longer one.
   - **Bias penalty:** a positive average deviation is added back to every token as a penalty.
3. **Phase B also checks that `decode(encode(text)) == text` exactly**, in a process that never saw the text.

Under these rules the score is a true upper bound on the text's code length, whatever your tokenizer does. Peeking at future tokens, skipping normalization, emitting unnormalized scores and lossy tokenizers are all rejected.

### Budgets

- **Time:** phase A has a wall-clock limit of 1.2× the baseline's scoring time on the validation and test documents. It is measured on the official GPU (RTX 5090) with the weights already loaded; weight loading has its own cap.
- **Compute:** a FLOPs-per-token probe runs under PyTorch's `FlopCounterMode` and is reported. A cap (1.2× the baseline) applies only when the baseline's FLOPs are observable; the vLLM baseline runs in a separate engine process the counter cannot see, so the current board has no FLOPs cap and the time limit is the budget.
- **Environment:** packages are exactly those in the pinned scoring image (`containers/arch.Dockerfile`). Nothing can be installed at scoring time.

Rejection codes: `load_timeout`, `time_limit_exceeded`, `audit_timeout`, `model_error`, `protocol_error`, `invalid_contract`, `invalid_tokens`, `invalid_claims`, `audit_failed`, `roundtrip_failed`, `flops_exceeded`. Output from your code is never returned, because it could contain private text.

## Workflow

```bash
delta model pull <baseline-version-id> --output ./qwen        # weights, config, tokenizer
delta bundle init ./mine --parent ./qwen                       # PyTorch template (--template vllm: the baseline)
# ... edit ./mine/delta_model.py, retrain or convert weights ...
delta bundle check ./mine --limit 200                          # same harness, FineWeb validation
delta bundle build ./mine --model 1b-arch
delta submit ./mine --title 'Sliding-window mixer'
```

`delta bundle check` runs the official harness in local subprocesses. It is not sandboxed and does not enforce isolation between phases; official scoring does. Pass `--time-limit` and `--flops-cap` to rehearse the budgets.
