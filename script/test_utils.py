from utils_ner import NerDataset, Split, get_labels
from transformers import BertTokenizer

def main():
    data_file_path = "./labels.txt"
    tokenizer = BertTokenizer.from_pretrained("./vocab.txt")
    labels = get_labels(data_file_path)
    print("Labels:", labels)

    dataset = NerDataset(
        data_dir="./../dataset/",
        tokenizer=tokenizer,
        labels=labels,
        max_seq_length=128,
        model_type="bert"
    )
    
    example = dataset[0]
    print("Example:", example)

    test_split = Split("test")
    print("Test Split:", test_split)

if __name__ == "__main__":
    main()
