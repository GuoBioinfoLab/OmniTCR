# Changelog

All notable changes to the public inference package are documented here.

## 1.0.0

- Added checkpoint-compatible PM, PT, PMT and PMAB binding inference.
- Added exact repertoire-level cancer-screening inference with top-1,000
  selection and median aggregation.
- Added SFT and PMI TCR generation using one shared `OmniTCR(SFT)` checkpoint.
- Added MHC allele-to-pseudosequence conversion and exact component tokens.
- Added Python and CSV APIs, a command-line interface, tests and examples.
- Aligned automatic checkpoint resolution with the published Hugging Face
  folder names, including `OmniTCR(FFT)_CA`.

