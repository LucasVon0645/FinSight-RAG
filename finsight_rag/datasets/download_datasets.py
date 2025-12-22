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


def download_rag_dataset():
    print("Downloading EDGAR Year 2020 dataset...")

    year_2020_test_dataset = load_dataset(
        "eloukas/edgar-corpus", "year_2020", split="test", trust_remote_code=True
    )
    year_2020_test_dataset.save_to_disk("data/rag/edgar-2020/test")

    metadata = year_2020_test_dataset.info
    print("\nDataset description:")
    print(metadata.description)
    print("\n")

    with open(
        "data/rag/edgar-2020/metadata.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(metadata.description)
        f.write("\n")
        f.write(metadata.homepage)

    year_2020_validate_dataset = load_dataset(
        "eloukas/edgar-corpus", "year_2020", split="validation", trust_remote_code=True
    )
    year_2020_validate_dataset.save_to_disk("data/rag/edgar-2020/validation")

    year_2020_train_dataset = load_dataset(
        "eloukas/edgar-corpus", "year_2020", split="train", trust_remote_code=True
    )
    year_2020_train_dataset.save_to_disk("data/rag/edgar-2020/train")


if __name__ == "__main__":
    download_rag_dataset()
