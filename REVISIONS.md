# Pinned model revisions

Every Hugging Face checkpoint used in the paper, pinned to the exact commit
revision resolved at run time. Pass these to `from_pretrained(..., revision=<sha>)`
to reproduce the exact weights. Abbreviated forms of these SHAs appear in
Appendix D of the paper.

| Checkpoint | Full commit revision |
|---|---|
| google/gemma-3-12b-it | 96b6f1eccf38110c56df3a15bffe176da04bfd80 |
| meta-llama/Llama-3.1-8B-Instruct | 0e9e39f249a16976918f6564b8830bc894c89659 |
| Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 |
| google/gemma-4-E2B-it | 70af34e20bd4b7a91f0de6b22675850c43922a03 |
| google/gemma-4-E4B-it | fee6332c1abaafb77f6f9624236c63aa2f1d0187 |
| google/gemma-4-12B-it | 5926caa4ec0cac5cbfadaf4077420520de1d5205 |
| google/gemma-4-31B-it | 3548789868c5356dbf307c98e6f609007b82b3eb |
| google/gemma-4-26B-A4B-it | 20da991ab4afab98e8f910c4a2e8f4fbefc404ad |
| google/diffusiongemma-26B-A4B-it | 0f28bc42f588fbd8f71e08102b1c3960298a1358 |
| Qwen/Qwen3.6-27B | 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 |
| meta-llama/Llama-3.1-70B-Instruct | 1605565b47bb9346c5515c34102e054115b4f98b |
| intfloat/e5-large-v2 | f169b11e22de13617baa190a028a32f3493550b6 |
| unitary/toxic-bert (Detoxify) | 4d6c22e74ba2fdd26bc4f7238f50766b045a0d94 |
| s-nlp/roberta_toxicity_classifier | 048c25bb1e199b98802784f96325f4840f22145d |
| tomh/toxigen_roberta | 0e65216a558feba4bb167d47e49f9a9e229de6ab |
| google/gemma-scope-2-12b-it | 4c419f1ba0be8b7754d4151d4f26c23b92a9029e |
| andyrdt/saes-llama-3.1-8b-instruct | 68226a9ff81811b89d93ad5c54064aae7927cc19 |
| andyrdt/saes-qwen2.5-7b-instruct | c37e53c4bb07127ad17ab88f28b93d4e87142e59 |

The two andyrdt SAE repos each ship four trainers at the same 131,072-wide
dictionary; the paper uses `trainer_2` (k=128 active latents) at residual
layers 19/23/27. The commit SHA pins the repo; the `trainer_2` subdirectory
pins the checkpoint.

Detoxify scoring runs through the `detoxify` PyPI package (version 0.5.2, as
pinned in `uv.lock`), whose `original` model resolves to the unitary/toxic-bert
checkpoint above. The package's own model card notes the raw HF checkpoint can
score slightly differently than the library path; the library path is the one
used throughout, so reproduce with `detoxify==0.5.2` rather than loading the
checkpoint directly.

## Upstream benchmark source (prompt and rule logic)

The prompted-LLM arm ports Kumar et al.'s released GPT-3.5 harness. The
upstream source of truth is the authors' `rule_based_moderation.py`
(Kumar, AbuHashem, Durumeric, *Watch Your Language: Investigating Content
Moderation with Large Language Models*, ICWSM 2024,
DOI `10.1609/icwsm.v18i1.31358`, arXiv `2309.14517`). Our port,
`pipeline/kumar_mod/kumar_data.py`, reproduces that script function by
function so the local-model replication is faithful by construction:

- `unmark()` reproduces the markdown stripping (markdown lib, plain output).
- `extract_rules_text()` reproduces the rule construction (skip
  `kind=='link'` rules, unmark, number from 1).
- The `FIRST/SECOND/THIRD/SAMPLE` prompt-turn strings are byte-for-byte the
  turns Kumar sent GPT-3.5, including the canned `"You are a stupid idiot."`
  few-shot prime and the 8-space indentation in the first user turn.

The only documented deviations are intentional: we target local instruct
models through a chat template instead of `openai.ChatCompletion`, and we
read the authors' released balanced CSVs. Nothing about the prompt content
or rule construction changes. The exact upstream commit revision will be
pinned here at camera-ready once obtained from the benchmark authors; until
then, the port can be audited against the prompt listing in the paper's
Appendix D.1, which is reproduced verbatim from these strings.
