# SignWritingCLIP

CLIP model to embed SignWriting images.

Used in [signwriting-evaluation](https://github.com/sign-language-processing/signwriting-evaluation).

## Training a model

```bash
# 0. Setup the environment.
conda create --name vq python=3.11
conda activate clip
pip install .

# 1. Creates a dataset of SignWriting images.
DATA_DIR=/scratch/amoryo/clip
sbatch scripts/create_dataset.sh "$DATA_DIR"

# 2. Trains the model and reports to `wandb`.
sbatch scripts/train_model.sh "$DATA_DIR"
```

## CLIP vs SignWritingCLIP

### Cosine Similarity Distribution

The original CLIP model encodes all SignWriting images as very similar embeddings,
as evident by the distribution of cosine similarities between embeddings of random signs.
The distribution is head-heavy, with most similarities being above 0.9.

![cosine similarity distribution for CLIP](assets/distribution/CLIP.png)

On the other hand, SignWritingCLIP encodes SignWriting images as more diverse embeddings.
This is evident by a more tail-heavy distribution.

![cosine similarity distribution for SignWritingCLIP](assets/distribution/SignWritingCLIP.png)

### Nearest Neighbors

For three signs representing the sign for "hello", the nearest neighbors are shown below.

<table style="text-align: center">
<thead>
<tr><td></td><td colspan='2'><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/ref.png' /></td><td colspan='2'><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/ref.png' /></td><td colspan='2'><img src='assets/matches/M520x520S14c20480x484S27106505x480/ref.png' /></td></tr>
<tr><td></td><td>CLIP</td><td>SignWritingCLIP</td><td>CLIP</td><td>SignWritingCLIP</td><td>CLIP</td><td>SignWritingCLIP</td></tr>
</thead>
<tbody>
<tr><td>1</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIP/0.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/SignWritingCLIP/0.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIP/0.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/SignWritingCLIP/0.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIP/0.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/SignWritingCLIP/0.png' /></td></tr>
<tr><td>2</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIP/1.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/SignWritingCLIP/1.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIP/1.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/SignWritingCLIP/1.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIP/1.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/SignWritingCLIP/1.png' /></td></tr>
<tr><td>3</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIP/2.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/SignWritingCLIP/2.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIP/2.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/SignWritingCLIP/2.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIP/2.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/SignWritingCLIP/2.png' /></td></tr>
<tr><td>4</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIP/3.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/SignWritingCLIP/3.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIP/3.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/SignWritingCLIP/3.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIP/3.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/SignWritingCLIP/3.png' /></td></tr>
<tr><td>5</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIP/4.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/SignWritingCLIP/4.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIP/4.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/SignWritingCLIP/4.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIP/4.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/SignWritingCLIP/4.png' /></td></tr>
<tr><td>6</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIP/5.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/SignWritingCLIP/5.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIP/5.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/SignWritingCLIP/5.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIP/5.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/SignWritingCLIP/5.png' /></td></tr>
<tr><td>7</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIP/6.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/SignWritingCLIP/6.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIP/6.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/SignWritingCLIP/6.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIP/6.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/SignWritingCLIP/6.png' /></td></tr>
<tr><td>8</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIP/7.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/SignWritingCLIP/7.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIP/7.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/SignWritingCLIP/7.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIP/7.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/SignWritingCLIP/7.png' /></td></tr>
<tr><td>9</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIP/8.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/SignWritingCLIP/8.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIP/8.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/SignWritingCLIP/8.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIP/8.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/SignWritingCLIP/8.png' /></td></tr>
<tr><td>10</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIP/9.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/SignWritingCLIP/9.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIP/9.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/SignWritingCLIP/9.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIP/9.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/SignWritingCLIP/9.png' /></td></tr>
</tbody>
</table>