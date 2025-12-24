from datasets import load_dataset


def download_ner_dataset():
    print("Downloading Finer-139 NER dataset...")
    dataset_dict = load_dataset("nlpaueb/finer-139", trust_remote_code=True)

    metadata = dataset_dict["train"].info
    print("\nDataset description:")
    print(metadata.description)
    print("\n")

    print("Dataset features:")
    print(metadata.features)
    print("\n")

    with open("data/ner/finer-139/metadata.txt", "w", encoding="utf-8") as f:
        f.write(metadata.description)
        f.write("\n")
        f.write(metadata.homepage)

    print("Saving dataset locally...")

    # Save locally in Hugging Face format
    dataset_dict["train"].save_to_disk("data/ner/finer-139/train")
    dataset_dict["validation"].save_to_disk("data/ner/finer-139/validation")
    dataset_dict["test"].save_to_disk("data/ner/finer-139/test")

def download_sentiment_analysis_dataset():
    print("Downloading Financial PhraseBank Sentiment Analysis dataset...")
    dataset = load_dataset("financial_phrasebank", "sentences_allagree", trust_remote_code=True)

    metadata = dataset["train"].info
    print("\nDataset description:")
    print(metadata.description)
    print("\n")

    print("Dataset features:")
    print(metadata.features)
    print("\n")

    with open(
        "data/sentiment_analysis/financial_phrasebank/metadata.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(metadata.description)
        f.write("\n")
        f.write(metadata.homepage)

    print("Saving dataset locally...")

    # Save locally in Hugging Face format
    dataset["train"].save_to_disk("data/sentiment_analysis/financial_phrasebank/train")


if __name__ == "__main__":
    download_sentiment_analysis_dataset()
