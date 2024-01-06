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
    with open(csv_path, 'r', encoding="utf-8") as f:
        csv.field_size_limit(2 ** 20)
        reader = csv.DictReader(f)
        for row in reader:
            sw, caption = row["source"], row["target"]

            # There is a bug in the training script that does not allow the value "None" or "null"
            if caption.strip().lower() in ["none", "null", "na"]:
                continue

            sw = remove_dollar_tokens(sw)
            signwritings.add(sw)
            data.append((sw, caption))

    # map signwriting to image path
    images = list(images_directory.glob("*.png"))
    image_names = set(image.stem for image in images)
    sw_to_image = {}
    for sw in signwritings:
        fsw_md5 = hashlib.md5(sw.encode('utf-8')).hexdigest()
        if fsw_md5 in image_names:
            sw_to_image[sw] = images_directory / f"{fsw_md5}.png"

    # filter out data without images
    data = [(sw, caption) for sw, caption in data if sw in sw_to_image]

    for i, (sw, caption) in enumerate(data):
        yield i, {
            "caption": caption,
            "image_path": sw_to_image[sw],
        }


def save_dataset_csv(dataset, csv_output_path: Path):
    with open(csv_output_path, 'w', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "caption"], quoting=csv.QUOTE_NONNUMERIC, quotechar='"')
        writer.writeheader()
        for i, example in dataset:
            writer.writerow({
                "image_path": example["image_path"],
                "caption": example["caption"],
            })


if __name__ == "__main__":
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
