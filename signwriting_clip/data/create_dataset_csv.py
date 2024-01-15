import argparse
import csv
import hashlib
from pathlib import Path


def remove_dollar_tokens(text: str):
    return " ".join(token for token in text.split(" ") if not token.startswith("$")).strip()


def generate_samples(images_directory: Path, csv_path: Path):
    # Load data from CSV file
    signwritings = set()
    data = []
    with open(csv_path, 'r', encoding="utf-8") as csv_file:
        csv.field_size_limit(2 ** 20)
        reader = csv.DictReader(csv_file)
        for row in reader:
            signwriting, caption = row["source"], row["target"]

            # There is a bug in the training script that does not allow the value "None" or "null"
            if caption.strip().lower() in ["none", "null", "na"]:
                continue

            signwriting = remove_dollar_tokens(signwriting)
            signwritings.add(signwriting)
            data.append((signwriting, caption))

    # map signwriting to image path
    images = list(images_directory.glob("*.png"))
    image_names = set(image.stem for image in images)
    signwriting_to_image = {}
    for signwriting in signwritings:
        fsignwriting_md5 = hashlib.md5(signwriting.encode('utf-8')).hexdigest()
        if fsignwriting_md5 in image_names:
            signwriting_to_image[signwriting] = images_directory / f"{fsignwriting_md5}.png"

    # filter out data without images
    data = [(signwriting, caption) for signwriting, caption in data if signwriting in signwriting_to_image]

    for i, (signwriting, caption) in enumerate(data):
        yield i, {
            "caption": caption,
            "image_path": signwriting_to_image[signwriting],
        }


def save_dataset_csv(dataset, csv_output_path: Path):
    with open(csv_output_path, 'w', encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "caption"],
                                quoting=csv.QUOTE_NONNUMERIC, quotechar='"')
        writer.writeheader()
        for _, example in dataset:
            writer.writerow({
                "image_path": example["image_path"],
                "caption": example["caption"],
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-directory", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    images_directory = Path(args.images_directory)
    output_path = Path(args.output_path)
    csv_path = Path(args.csv)

    dataset = generate_samples(images_directory, csv_path)
    save_dataset_csv(dataset, output_path)


if __name__ == "__main__":
    main()
