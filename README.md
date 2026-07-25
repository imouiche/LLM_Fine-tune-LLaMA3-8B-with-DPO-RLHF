# LLaMA-3-8B Fine-Tuning with Direct Preference Optimization (DPO)

This repository contains a notebook for fine-tuning **`unsloth/llama-3-8b`** using **Direct Preference Optimization (DPO)** with LoRA adapters, trained on the [`Intel/orca_dpo_pairs`](https://huggingface.co/datasets/Intel/orca_dpo_pairs) preference dataset.

The goal is to align a base (non-instruction-tuned) language model toward more helpful, human-preferred responses using pairwise preference data, without requiring full-parameter fine-tuning or a separate reward model (as RLHF/PPO would).

---

## Two Approaches to DPO

There are generally two common pipelines for getting a model to a preference-aligned end state with DPO, and it's worth being explicit about which one this notebook implements:

1. **SFT → DPO (two-stage pipeline).** First supervised-fine-tune a base model on instruction/response pairs to teach it conversational structure and task-following behavior, *then* apply DPO on top of the resulting instruction-tuned checkpoint to sharpen preference alignment. This is the approach used in my **Gate-DPO** paper [`link`](https://arxiv.org/abs/2605.02626), where the SFT stage establishes baseline instruction-following before DPO refines response preference.

2. **Direct DPO on a base model (single-stage, this repository).** Apply DPO straight to a base model that has *not* been instruction-tuned, using a manually-defined chat template (ChatML-style, in this case) to impose conversational structure on the fly. This skips the separate SFT stage entirely and relies on the preference dataset itself (paired with an explicit prompt template) to teach both structure and preference simultaneously.

This notebook demonstrates approach **(2)** — direct DPO applied to `unsloth/llama-3-8b`, a base model. This is a useful contrast case to the SFT→DPO pipeline in Gate-DPO: it's faster to set up (one training stage instead of two) but places more weight on the preference dataset to also teach the model conversational formatting, which is typically harder to learn well from preference data alone compared to a dedicated SFT stage.

---

## What's in This Notebook

| Step | Description |
|---|---|
| 1. Dataset formatting | Loads `Intel/orca_dpo_pairs` and reformats each example into `(prompt, chosen, rejected)` triples using a manually-defined ChatML template |
| 2. Base model sanity check | Loads the base model in fp16 and runs a quick generation test to confirm chat-template formatting and inference are wired up correctly, before training |
| 3. LoRA configuration | Defines a `LoraConfig` targeting the attention and MLP projection layers |
| 4. DPO training | Configures training arguments (batch size, LR schedule, precision, etc.) and trains with `DPOTrainer` from `trl` |

---

## Why a Manual Chat Template?

`unsloth/llama-3-8b` is a **base** model, not an instruct/chat model; its tokenizer ships without a `chat_template`. Since this notebook takes the direct-DPO-on-base-model route, the notebook manually defines a ChatML-style template (`<|im_start|>`/`<|im_end|>` tokens) and assigns it to `tokenizer.chat_template` before formatting the dataset. This is what allows `tokenizer.apply_chat_template(...)` to work at all — without it, the base tokenizer has no notion of turn structure.

If you plan to add `<|im_start|>`/`<|im_end|>` as genuine special tokens (recommended for cleaner tokenization), remember to also resize the model's embedding matrix:

```python
tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
model.resize_token_embeddings(len(tokenizer))
```

---

## Requirements

```
torch
transformers
datasets
peft
trl
bitsandbytes
accelerate
evaluate
```

Tested on Google Colab with a single GPU. Precision is set to **fp16** rather than **bf16** to support older (pre-Ampere) GPUs that lack native bf16 support; switch to `bf16=True` / `fp16=False` if you're running on an **A100/H100** or similar.

> **Note:** `trl`'s `DPOTrainer` API has changed across versions; in newer releases, arguments like `beta`, `max_length`, and `max_prompt_length` live on a `DPOConfig` object rather than being passed directly to `DPOTrainer`. Check `pip show trl` and adjust accordingly if you hit a `TypeError` for unexpected keyword arguments.

---

## Experiment Configuration (Summary)

| Parameter | Value | Notes |
|---|---|---|
| LoRA rank (`r`) | 16 | |
| LoRA alpha | 32 | Effective scale = alpha / r |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | |
| Effective batch size | 4 | `per_device_train_batch_size=1` × `gradient_accumulation_steps=4` |
| Learning rate | 5e-5 | Cosine schedule, 50 warmup steps |
| Precision | bp16/fp16 | |
| DPO beta | 0.1 | Controls divergence from the frozen reference policy |
| Max prompt / total length | 1024 / 1536 | |

---

## Related Work

- [`Gate-DPO`](https://arxiv.org/abs/2605.02626); implements the SFT → DPO two-stage pipeline referenced above.

---

## License
 Apache 2.0
