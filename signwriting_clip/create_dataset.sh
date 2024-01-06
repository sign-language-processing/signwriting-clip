#!/bin/bash

#SBATCH --job-name=create-dataset
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=dataset.out

#SBATCH --ntasks=1

set -e # exit on error
set -x # echo commands

module load anaconda3
source activate clip

# Download the SignBank repository if not exists
SIGNBANK_DIR="/home/amoryo/sign-language/signbank-annotation/signbank-plus"
[ ! -d "$SIGNBANK_DIR" ] && \
git clone https://github.com/sign-language-processing/signbank-plus.git "$SIGNBANK_DIR"

# Process data for machine translation if not exists
[ ! -d "$SIGNBANK_DIR/data/parallel/cleaned" ] && \
python "$SIGNBANK_DIR/signbank_plus/prep_nmt.py"

# Create images from all written signs (can take a long time)
PROCESSED_DATA_DIR="/scratch/$(whoami)/SignWritingImages"
mkdir -p $PROCESSED_DATA_DIR

conda install nodejs -y

for name in "raw" "sign2mint" "signsuisse"
do
  # TODO fix error: reimplement with python https://github.com/sign-language-processing/signwriting/issues/1
  python "$SIGNBANK_DIR/signbank_plus/create_signwriting_images.py" \
      --input-path="$SIGNBANK_DIR/data/$name.csv" \
      --output-path="$PROCESSED_DATA_DIR"
done


# For CLIP, images should be image_size = 224. We should pad small images to 224x224, and remove larger images
pip install Pillow
# This takes about 2.5 hours for 240k images
python data/preprocess_images_for_clip.py \
    --input-path="$PROCESSED_DATA_DIR" \
    --output-path="$PROCESSED_DATA_DIR-clip"

# Convert to huggingface dataset
HF_DATASET_CSV="/scratch/$(whoami)/sign-writing-clip.csv"

[ ! -f "$HF_DATASET_CSV" ] && \
python data/create_dataset_csv.py \
  --images-directory="$PROCESSED_DATA_DIR-clip" \
  --csv="$SIGNBANK_DIR/data/parallel/cleaned/train.csv" \
  --output-path="$HF_DATASET_CSV"
