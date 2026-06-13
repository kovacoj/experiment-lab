from transformers import pipeline


MODEL_DIR = ".models/sentiment/cardiffnlp-twitter-xlm-roberta-base-sentiment"


def main() -> None:
    sentiment = pipeline(
        task="sentiment-analysis",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        device=-1,
        truncation=True,
    )

    texts = [
        "Great coffee and fast service.",
        "We waited 20 minutes and the service was slow.",
        "Coffee shop in Prague.",
    ]

    results = sentiment(texts, batch_size=4)

    for text, result in zip(texts, results):
        print("---")
        print(text)
        print(result)


if __name__ == "__main__":
    main()
