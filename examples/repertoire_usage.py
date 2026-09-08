from omnitcr import OmniTCR


def main():
    model = OmniTCR(task="repertoire", device="cuda")

    score = model.predict_repertoire(
        trb_sequences=[
            "CARSVGGNGGNTEAFF",
            "CARSVGGNGGNTEAFF",
            "CARSVGANGGNTEAFF",
        ],
        weights=[9, 4, 2],
    )
    print(f"Repertoire score: {score:.6f}")

    model.predict_csv(
        input_path="examples/data/repertoire_examples.csv",
        output_path="repertoire_scores.csv",
    )


if __name__ == "__main__":
    main()
