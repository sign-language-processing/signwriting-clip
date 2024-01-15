import argparse
import csv
import hashlib
from csv import DictReader

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from signwriting.visualizer.visualize import signwriting_to_image


def process_image_clip(fsw: str, dst_path: Path):
    # Create a 224x224 RGB image with a white background
    size = 224
    img = signwriting_to_image(fsw)  # this is RGBA

    if img.width > size or img.height > size:
        return

    new_img = Image.new('RGB', (size, size), (255, 255, 255))

    # Calculate the position to paste the image so that it's centered
    x_offset = (size - img.width) // 2
    y_offset = (size - img.height) // 2
    offset = (x_offset, y_offset)

    # Paste the output_im image onto the white background
    new_img.paste(img, offset, img)

    new_img.save(dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_path = Path(args.output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    existing_files = set(map(str, output_path.glob('*.png')))
    print(f"Found {len(existing_files)} existing files.")

    with open(input_csv, 'r', encoding="utf-8") as csv_f:
        csv.field_size_limit(2 ** 20)  # Increase limit to 1MB (2^20 characters)
        reader = DictReader(csv_f)
        unique_fsw = set(row["sign_writing"] for row in reader)

    for fsw in tqdm(unique_fsw):
        fsw_md5 = hashlib.md5(fsw.encode('utf-8')).hexdigest()
        output_file = output_path / f"{fsw_md5}.png"
        if str(output_file) not in existing_files:
            try:
                process_image_clip(fsw, output_file)
            except Exception as exception:  # pylint: disable=broad-except
                print(f"Failed to process {fsw}: {exception}")


if __name__ == "__main__":
    main()
