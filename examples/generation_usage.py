from omnitcr import OmniTCR


def main():
    sft = OmniTCR(task="generation", mode="sft", device="cuda")
    sft_sequences = sft.generate(
        peptide="GADGVGKSA",
        mhc="HLA-A*01:01",
        num_sequences=100,
    )
    print(sft_sequences)

    pmi = OmniTCR(task="generation", mode="pmi", device="cuda")
    pmi.generate_csv(
        input_path="examples/data/generation_examples.csv",
        output_path="pmi_generated_tcrs.csv",
        num_sequences=100,
    )


if __name__ == "__main__":
    main()
