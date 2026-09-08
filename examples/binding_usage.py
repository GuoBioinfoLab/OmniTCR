from omnitcr import OmniTCR


def main():
    model = OmniTCR(task="pmt", device="cuda")

    score = model.predict(
        peptide="LLQCTQQAV",
        mhc="HLA-A*01:01",
        trb="CASSQDRGIGYGYTF",
    )
    print(f"Binding score: {score:.6f}")

    model.predict_csv(
        input_path="examples/data/pmt_examples.csv",
        output_path="pmt_scores.csv",
    )


if __name__ == "__main__":
    main()
