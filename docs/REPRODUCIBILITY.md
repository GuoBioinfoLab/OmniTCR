# Reproducibility contract

This package exposes inference only. It intentionally preserves the model
architectures, token order, input serialization and post-processing used by
the manuscript evaluation scripts.

## Binding prediction

- The checkpoint contains the full decoder backbone and the binary head.
- The representation is the hidden state at the final closing component token,
  immediately before `[EOS]`.
- The head is `Linear(768, 384) -> GELU -> Dropout(0.1) -> Linear(384, 1)`.
- The returned score is `sigmoid(logit)`.

## Repertoire-level cancer screening

- Input CDR3beta sequences are raw strings; `[TRB]` tokens are added internally.
- Duplicate rows are retained.
- When weights are present, rows are sorted by decreasing weight and the top
  1,000 rows per sample are retained. Weights are then normalized exactly as in
  the evaluation script, although median aggregation does not use them.
- The head is `Linear(768, 1024) -> ReLU -> Linear(1024, 2)`.
- Sequence scores are class-1 softmax probabilities.
- The reported sample score is the median sequence score.

## SFT generation

- Prompt: `[EPI]peptide[EPI][HLA]pseudosequence[HLA][TRB]`, preceded by BOS.
- Beam search: 400 beams, at most 40 new tokens, no sampling,
  `repetition_penalty=0.7`, `length_penalty=0.6`, early stopping enabled.
- Returned sequences retain beam order and pass the manuscript SFT validity
  filter: length 7--24, starts with `C`, ends with `F` or `W`.
- SFT outputs are not deduplicated or refilled after filtering.

## PMI generation

- The same SFT checkpoint produces and scores all candidates.
- The first 200 beams are filtered to canonical amino-acid CDR3beta sequences
  and deduplicated while retaining first-seen order.
- Each candidate continuation is scored as `CDR3beta[TRB]`, without EOS.
- The target score and null score are average continuation log probabilities.
- Null prompt: `[EPI][EPI][HLA][HLA][TRB]`.
- Ranking score: `target_avg_logp - 0.8 * null_avg_logp`.
- Ties are resolved by target log probability and then original beam order.
- The public API returns only generated CDR3beta sequences.

## Attention masks

Attention masks are passed for every classification, generation and PMI-scoring
forward pass. Padded positions are zero and biological tokens are one.

## Checkpoint versioning

For a published analysis, pass an immutable Hugging Face commit or release tag
through `revision`. Record the code version, model revision, MHC table version,
hardware and output CSV checksums alongside the analysis.

