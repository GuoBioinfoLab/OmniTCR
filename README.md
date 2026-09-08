# OmniTCR

OmniTCR is an inference package for T-cell receptor (TCR) and
peptide--major histocompatibility complex (pMHC) modeling. It provides one
consistent Python and command-line interface for:

- peptide--MHC and peptide--TCR binding prediction;
- paired and unpaired TCR--pMHC binding prediction;
- repertoire-level cancer-associated scoring; and
- pMHC-conditioned TCRbeta CDR3 generation using SFT or PMI reranking.

This repository contains inference code only. It does not include model
training, benchmark construction, threshold selection, bootstrap confidence
intervals or manuscript figure generation.

## Supported tasks

| API task | Biological input | Hugging Face checkpoint | Public output |
| --- | --- | --- | --- |
| `pm` | peptide + MHC allele | `OmniTCR(FFT)_PM` | one score per row |
| `pt` | peptide + TRB CDR3 | `OmniTCR(FFT)_PT` | one score per row |
| `pmt` | peptide + MHC allele + TRB CDR3 | `OmniTCR(FFT)_PMT` | one score per row |
| `pmab` | peptide + MHC allele + TRA/TRB CDR3 | `OmniTCR(FFT)_PMAB` | one score per row |
| `repertoire` | TRB CDR3 repertoire | `OmniTCR(FFT)_CA` | one score per sample |
| `generation` | peptide + MHC allele | `OmniTCR(SFT)` | generated TRB CDR3 sequences |

SFT and PMI generation use the same `OmniTCR(SFT)` checkpoint. PMI is an
inference-time reranking procedure, not a separately trained model.

## Repository structure

```text
OmniTCR/
├── src/omnitcr/               # installable inference package
│   ├── api.py                 # public Python API
│   ├── checkpoints.py         # Hugging Face and local model loading
│   ├── cli.py                 # command-line interface
│   ├── datasets.py            # inference datasets and dynamic padding
│   ├── generation.py          # SFT and PMI generation
│   ├── modeling.py            # checkpoint-compatible architectures
│   ├── preprocessing.py       # validation and input serialization
│   ├── tokenizer.py           # fixed 34-token tokenizer
│   └── resources/             # MHC pseudosequence lookup
├── examples/                  # Python and CSV examples
├── tests/                     # unit and inference-contract tests
├── scripts/                   # checkpoint maintenance utilities
├── docs/                      # detailed reproducibility documentation
├── environment.yaml           # minimal manuscript-compatible environment
├── environment-full.yaml      # complete author environment export
└── pyproject.toml             # package and CLI definition
```

## Installation

### Conda environment used for inference

The minimal environment preserves the core versions from the supplied
manuscript environment and installs the CUDA 12.1 build of PyTorch:

```bash
conda env create -f environment.yaml
conda activate omnitcr
python -m pip install -e .
```

`environment-full.yaml` is the complete pinned export of the original `ft`
environment, with only its machine-specific prefix removed. It is retained for
forensic reproducibility but includes training and analysis packages that are
not required by the inference API.

For a CPU-only installation, create a Python 3.10--3.12 environment, install an
appropriate CPU build of PyTorch, and then run:

```bash
python -m pip install -e .
```

### Hugging Face authentication

The default model repository is `loveCloud/OmniTCR`. While it is private,
authenticate once before running inference:

```bash
hf auth login
```

The API then downloads only the files needed for the selected task. For a
published analysis, pass a release tag or immutable commit through `revision`
rather than relying on the moving `main` branch.

## Quick start: binding prediction

### One input

```python
from omnitcr import OmniTCR

model = OmniTCR(
    task="pmt",
    device="cuda",
)

score = model.predict(
    peptide="LLQCTQQAV",
    mhc="HLA-A*01:01",
    trb="CASSQDRGIGYGYTF",
)

print(score)
```

`score` is the sigmoid-transformed class-1 model output. It is a continuous
model score, not a prespecified binary decision and not necessarily a
calibrated binding probability.

### In-memory batch

```python
scores = model.predict_batch(
    [
        {
            "peptide": "LLQCTQQAV",
            "mhc": "HLA-A*01:01",
            "trb": "CASSQDRGIGYGYTF",
        },
        {
            "peptide": "CLLGTYTQDV",
            "mhc": "HLA-A*01:02",
            "trb": "CSAPGQSRGYTF",
        },
    ],
    batch_size=128,
)
```

### CSV input

```python
model.predict_csv(
    input_path="examples/data/pmt_examples.csv",
    output_path="pmt_scores.csv",
    batch_size=128,
)
```

The output preserves every input row and column and appends `score`. Row order
is unchanged. Required columns are case-insensitive but the following names and
order are recommended:

| Task | Required CSV columns |
| --- | --- |
| `pm` | `peptide,mhc` |
| `pt` | `peptide,trb` |
| `pmt` | `peptide,mhc,trb` |
| `pmab` | `peptide,mhc,tra,trb` |

Public MHC input is always an allele such as `HLA-A*01:01`, not a
pseudosequence. Peptide, TRA and TRB inputs are raw amino-acid sequences and
must not contain `[EPI]`, `[HLA]`, `[TRA]` or `[TRB]` tokens.

## Quick start: repertoire-level cancer scoring

The repertoire API accepts raw TRB CDR3 sequences. Component tokens are added
internally.

```python
from omnitcr import OmniTCR

model = OmniTCR(
    task="repertoire",
    device="cuda",
)

score = model.predict_repertoire(
    trb_sequences=[
        "CARSVGGNGGNTEAFF",
        "CARSVGGNGGNTEAFF",
        "CARSVGANGGNTEAFF",
    ],
    weights=[9, 4, 2],
)

print(score)
```

CSV input uses one row per repertoire sequence:

```csv
trb,sample_id,weight
CARSVGGNGGNTEAFF,sample_1,9
CARSVGGNGGNTEAFF,sample_1,4
CARSVGANGGNTEAFF,sample_1,2
```

```python
model.predict_repertoire_csv(
    input_path="examples/data/repertoire_examples.csv",
    output_path="repertoire_scores.csv",
    batch_size=256,
    top_k=1000,
)
```

Output:

```csv
sample_id,score
sample_1,0.7342
```

The example score above illustrates the file schema only; it is not a bundled
model result.

This mode reproduces the manuscript evaluation logic exactly:

1. duplicate sequence rows are retained;
2. if weights are supplied, rows are sorted within each sample by decreasing
   weight and the top 1,000 are retained;
3. weights are normalized within each sample, as in the original dataset code;
4. each raw sequence is serialized as `[TRB]sequence[TRB]`;
5. the sequence score is `softmax(logits)[:, 1]`; and
6. the sample score is the median of the retained sequence scores.

Weights therefore determine top-1,000 selection but do not otherwise alter the
median. When no weight column is present, all rows are retained, matching the
original evaluation script.

The cancer-associated score is intended for research reproduction only and is
not a clinical diagnosis, screening recommendation or medical device output.

## Quick start: TCR generation

### SFT generation

```python
from omnitcr import OmniTCR

model = OmniTCR(
    task="generation",
    mode="sft",
    device="cuda",
)

tcrs = model.generate(
    peptide="GADGVGKSA",
    mhc="HLA-A*01:01",
    num_sequences=100,
)
```

### PMI generation

```python
model = OmniTCR(
    task="generation",
    mode="pmi",
    device="cuda",
)

tcrs = model.generate(
    peptide="GADGVGKSA",
    mhc="HLA-A*01:01",
    num_sequences=100,
)
```

Both calls return `list[str]`. PMI returns only the selected TCR sequences; PMI
scores, target/null likelihoods and beam metadata are intentionally omitted.

CSV input:

```csv
peptide,mhc
GADGVGKSA,HLA-A*01:01
HQNPVTGLLL,HLA-A*01:02
APARLERRHSA,HLA-A*01:03
```

`epitope` is accepted as an alias for the generation `peptide` column.

```python
model.generate_csv(
    input_path="examples/data/generation_examples.csv",
    output_path="generated_tcrs.csv",
    num_sequences=100,
)
```

Output columns:

```csv
peptide,mhc,rank,generated_trb
GADGVGKSA,HLA-A*01:01,1,CASSLGRASNQPQHF
```

The input identity and rank are retained only to associate each output sequence
with its pMHC condition.

### Exact generation behavior

The allele is converted to a pseudosequence and the prompt is constructed as:

```text
[BOS][EPI]peptide[EPI][HLA]pseudosequence[HLA][TRB]
```

SFT uses 400-beam deterministic generation and filters outputs to lengths
7--24 that begin with `C` and end with `F` or `W`. It preserves beam order,
duplicates and the original behavior of returning fewer than the requested
number if some beams fail the filter.

PMI generates the first 200 beam candidates, applies the stricter canonical
amino-acid filter, removes duplicates, and ranks candidates using:

```text
mean log P(TCR[TRB] | target pMHC)
    - 0.8 * mean log P(TCR[TRB] | empty pMHC)
```

The empty-pMHC prompt is `[EPI][EPI][HLA][HLA][TRB]`. No EOS token is included
in the scored continuation. See `docs/REPRODUCIBILITY.md` for the complete
contract.

## Command-line interface

After installation, all CSV workflows are available through `omnitcr`:

```bash
omnitcr binding \
  --task pmt \
  --input examples/data/pmt_examples.csv \
  --output pmt_scores.csv \
  --device cuda

omnitcr repertoire \
  --input examples/data/repertoire_examples.csv \
  --output repertoire_scores.csv \
  --top-k 1000 \
  --device cuda

omnitcr generate \
  --mode pmi \
  --input examples/data/generation_examples.csv \
  --output generated_tcrs.csv \
  --num-sequences 100 \
  --device cuda
```

Run `omnitcr --help` or `omnitcr <command> --help` for all options.

## Internal serialization

The public API constructs the exact model strings below:

```text
PM   : [EPI]peptide[EPI][HLA]pseudosequence[HLA]
PT   : [EPI]peptide[EPI][TRB]trb[TRB]
PMT  : [EPI]peptide[EPI][HLA]pseudosequence[HLA][TRB]trb[TRB]
PMAB : [EPI]peptide[EPI][HLA]pseudosequence[HLA][TRA]tra[TRA][TRB]trb[TRB]
CA   : [TRB]trb[TRB]
```

For classification, the tokenizer adds `[BOS]` and `[EOS]`, and the model uses
the hidden state of the closing component token immediately before `[EOS]`.
Attention masks are explicitly passed during binding prediction, repertoire
prediction, beam generation and PMI continuation scoring.

## Hugging Face model layout

Automatic loading expects the following case-sensitive structure:

```text
loveCloud/OmniTCR
├── OmniTCR(Base)/
│   └── config.json
├── OmniTCR(FFT)_CA/
│   └── model.safetensors
├── OmniTCR(FFT)_PM/
│   └── model.safetensors
├── OmniTCR(FFT)_PT/
│   └── model.safetensors
├── OmniTCR(FFT)_PMT/
│   └── model.safetensors
├── OmniTCR(FFT)_PMAB/
│   └── model.safetensors
└── OmniTCR(SFT)/
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors
    └── tokenizer files
```

The model files are deliberately excluded from this Git repository. The full
fine-tuning safetensors files must contain the complete `model_state_dict`,
including backbone and classifier parameters.

You can verify the remote layout after authentication:

```bash
python scripts/validate_hf_layout.py
```

## Local checkpoints

Local files can be used without changing inference behavior:

```python
model = OmniTCR(
    task="pmt",
    pretrained_model_path="/path/to/OmniTCR(Base)",
    checkpoint_path="/path/to/OmniTCR(FFT)_PMT/model.safetensors",
    mhc_pseudosequences_path="/path/to/NetMHCPan_MHC_pseudoseqs.csv",
    device="cuda",
)
```

For generation, `checkpoint_path` may point to the `OmniTCR(SFT)` directory.
For any task, a local copy of the full Hugging Face repository root is also
accepted.

To convert an original `.pt` checkpoint containing `model_state_dict`:

```bash
python scripts/convert_pt_checkpoint.py \
  checkpoint_epoch_3.pt \
  model.safetensors
```

The converter removes a leading DistributedDataParallel `module.` prefix and
does not write optimizer, scheduler or random-number-generator states.

## MHC pseudosequences

The mapper accepts either of these schemas:

```csv
mhc,pseudosequence
HLA-A*01:01,YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY
```

or the original schema:

```csv
MHC,label
HLA-A*01:01,YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY
```

Important: the bundled table contains only the eight example rows supplied
during package preparation. Before public release, replace
`src/omnitcr/resources/mhc_pseudosequences.csv` with the complete, exact table
used in the manuscript and record its source/version. Until then, pass the full
table with `mhc_pseudosequences_path` for alleles outside the bundled subset.

## Testing

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The test suite checks token IDs, component serialization, MHC conversion,
attention masks, checkpoint folder names, repertoire row handling, SFT
filtering and PMI ranking behavior. Numerical checkpoint tests should be run on
the authors' GPU server because model weights are not included in Git.

## Example data

Files under `examples/data/` demonstrate accepted schemas and software usage.
They are not benchmark datasets and should not be interpreted as experimentally
validated interaction records or clinical examples.

## Citation

Please cite the OmniTCR manuscript and archived software release once the
article and release DOI are available. Add the final citation and Zenodo DOI to
this section before publication.

## License

No open-source license has yet been specified in the materials provided for
this package. The repository owner should add the intended software license
before inviting third-party reuse.
