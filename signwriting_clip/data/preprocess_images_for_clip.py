import argparse

from pathlib import Path
from PIL import Image
from tqdm import tqdm


def process_image_clip(src_path: Path, dst_path: Path):
    # Create a 224x224 RGB image with a white background
    size = 224
    img = Image.open(src_path)  # this is RGBA

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
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    existing_files = set(map(str, output_path.glob('*.png')))
    print(f"Found {len(existing_files)} existing files.")

    for file in tqdm(list(input_path.glob('*.png'))):
        dst_path = output_path / file.name
        if dst_path in existing_files:
            continue

        process_image_clip(file, dst_path)


if __name__ == "__main__":
    main()
