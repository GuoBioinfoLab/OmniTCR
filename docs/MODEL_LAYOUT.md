# Model repository layout

The default model repository is `loveCloud/OmniTCR`. Automatic loading expects
the following case-sensitive paths:

```text
OmniTCR(Base)/config.json
OmniTCR(FFT)_CA/model.safetensors
OmniTCR(FFT)_PM/model.safetensors
OmniTCR(FFT)_PT/model.safetensors
OmniTCR(FFT)_PMT/model.safetensors
OmniTCR(FFT)_PMAB/model.safetensors
OmniTCR(SFT)/config.json
OmniTCR(SFT)/generation_config.json
OmniTCR(SFT)/model.safetensors
```

`OmniTCR(SFT)` may also contain tokenizer files. The public code constructs the
fixed 34-token tokenizer programmatically because this reproduces the original
evaluation scripts and protects token-ID compatibility.

For local inference, either pass the exact task directory/file or pass a local
copy of the model-repository root. The loader resolves both forms.

