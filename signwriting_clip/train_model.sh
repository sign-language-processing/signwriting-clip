#!/bin/bash

#SBATCH --job-name=train-clip-hf
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=train.out

#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --constraint=GPUMEM80GB|GPUMEM32GB

set -e # exit on error
set -x # echo commands

module load gpu
module load cuda

module load anaconda3
source activate clip

# Download transformers repository if not exists
[ ! -d "transformers" ] && \
git clone https://github.com/huggingface/transformers.git

# Install transformers from source
#pip install ./transformers accelerate huggingface-hub wandb
#pip install -r transformers/examples/pytorch/contrastive-image-text/requirements.txt

huggingface-cli login --token $HUGGINGFACE_TOKEN

OUTPUT_DIR="/scratch/$(whoami)/models/signwriting-clip/clip"
mkdir -p $OUTPUT_DIR

# Remove dataset filtering from `run_clip.py`
TRAIN_SCRIPT="transformers/examples/pytorch/contrastive-image-text/run_clip.py"
sed -i '/train_dataset\.filter(/,+2 s/^/#/' "$TRAIN_SCRIPT"

export WANDB_PROJECT="signwriting-clip"

# Train the model
python "$TRAIN_SCRIPT" \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path "openai/clip-vit-base-patch32" \
    --max_seq_length 77 \
    --freeze_text_model \
    --train_file "/data/$(whoami)/SignWritingImages/sign-writing-clip.csv" \
    --image_column "image_path" \
    --caption_column "caption" \
    --remove_unused_columns=False \
    --do_train \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --num_train_epochs="200" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --report_to "wandb" \
    --push_to_hub \
    --hub_model_id "sign/signwriting-clip"


# sbatch train.sh
# srun --pty -n 1 -c 1 --time=01:00:00 --gres=gpu:1 --constraint=GPUMEM80GB --mem=32G bash -l
# srun --pty -n 1 -c 1 --time=01:00:00 --mem=32G bash -l