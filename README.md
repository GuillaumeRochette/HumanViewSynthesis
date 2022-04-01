# Human View Synthesis

---
## Human Pose Manipulation and Novel View Synthesis using Differentiable Rendering

Guillaume Rochette, Chris Russell, Richard Bowden \
https://arxiv.org/abs/2111.12731

### Abstract

*We present a new approach for synthesizing novel views of people in new poses. Our novel differentiable renderer enables the synthesis of highly realistic images from any viewpoint. Rather than operating over mesh-based structures, our renderer makes use of diffuse Gaussian primitives that directly represent the underlying skeletal structure of a human. Rendering these primitives gives results in a high-dimensional latent image, which is then transformed into an RGB image by a decoder network. The formulation gives rise to a fully differentiable framework that can be trained end-to-end. We demonstrate the effectiveness of our approach to image reconstruction on both the Human3.6M and Panoptic Studio datasets. We show how our approach can be used for motion transfer between individuals; novel view synthesis of individuals captured from just a single camera; to synthesize individuals from any virtual viewpoint; and to re-render people in novel poses.*

### Videos

- View Synthesis from an Existing Viewpoint: https://youtu.be/vVCQM4cNwz8
- View Synthesis from a Virtual Viewpoint: https://youtu.be/gI9I8iLGVr4
- Motion Transfer between Actors: https://youtu.be/sbY34nnDy2M
- Paper Presentation (12'): https://youtu.be/Sxwpsi9TDOs

---
## Table of Contents

1. [Requirements](#requirements)
2. [Preparing Datasets](#preparing-datasets)
3. [Training](#training)
4. [Inference](#inference)
5. [Pre-trained Models](#pre-trained-models)
6. [Visualise the Renderer's Output](#visualise-the-renderers-output)
7. [Citation](#citation)

---
## Requirements

1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Set-up the following environment:
```
conda env create -f environment.yml -n my-env
conda activate my-env
export PYTHONPATH=.
```

## Preparing Datasets

Please follow the instructions detailed in each repositories:
- Panoptic Studio: https://github.com/GuillaumeRochette/PanopticProcessing
- Human3.6M: https://github.com/GuillaumeRochette/Human36MProcessing

## Training

If you want to train the model on `Panoptic`, at a resolution of `256x256`, using the `LPIPS` loss and then fine-tuning using the `Adaptive` adversarial framework:

1. Ensure you update the various `config.json` files, as the `root` is set to `/path/to/datasets/Panoptic`.
2. First pre-train the 2D-to-3D pose regression model:
```
python training/pose_2d_to_pose_3d.py \
    --hparams experiments/Pose2DToPose3D/Panoptic/hparams.json \
    --config experiments/Pose2DToPose3D/Panoptic/config.json
```
2. Pick the best model from `experiments/Pose2DToPose3D/Panoptic/lightning_logs/version_0/checkpoints` and save it under `experiments/Pose2DToPose3D/Panoptic/model.ckpt`.
3. Transfer the pre-trained weights to the novel view synthesis model:
```
python training/prepare_novel_view_synthesis.py \
    --config_pose_2d_to_pose_3d experiments/Pose2DToPose3D/Panoptic/config.json \
    --ckpt_pose_2d_to_pose_3d experiments/Pose2DToPose3D/Panoptic/model.ckpt \
    --hparams_novel_view_synthesis experiments/NovelViewSynthesis/Panoptic/256x256/LPIPS/hparams.json \
    --ckpt_novel_view_synthesis experiments/NovelViewSynthesis/Panoptic/256x256/LPIPS/initial.ckpt
```
4. Train the novel view synthesis model with the LPIPS loss:
```
python training/novel_view_synthesis.py \
    --hparams experiments/NovelViewSynthesis/Panoptic/256x256/LPIPS/hparams.json \
    --config experiments/NovelViewSynthesis/Panoptic/256x256/LPIPS/config.json \
    --ckpt experiments/NovelViewSynthesis/Panoptic/256x256/LPIPS/initial.ckpt
```
5. Pick the best model from `experiments/NovelViewSynthesis/Panoptic/256x256/LPIPS/lightning_logs/version_0/checkpoints` and save it under `experiments/NovelViewSynthesis/Panoptic/256x256/LPIPS/model.ckpt`.
6. Fine-tune the model with an adversarial loss:
```
python training/novel_view_synthesis.py \
    --hparams experiments/NovelViewSynthesis/Panoptic/256x256/Adaptive/hparams.json \
    --config experiments/NovelViewSynthesis/Panoptic/256x256/Adaptive/hparams.json \
    --ckpt experiments/NovelViewSynthesis/Panoptic/256x256/LPIPS/model.ckpt
```
7. Pick the best model from `experiments/NovelViewSynthesis/Panoptic/256x256/Adaptive/lightning_logs/version_0/checkpoints` and save it under `experiments/NovelViewSynthesis/Panoptic/256x256/Adaptive/model.ckpt`.

## Inference
- View Synthesis from an Existing Viewpoint:
```
python inference/infer_real.py \
    --ckpt experiments/NovelViewSynthesis/Panoptic/256x256/LPIPS/model.ckpt \
    --root /path/to/datasets/Panoptic \
    --sequence 171026_pose1/Subsequences/0 \
    --in_view 00 \
    --out_view 24 \
    --interval 2922 3751 \
    --output_dir experiments/NovelViewSynthesis/Panoptic/256x256/LPIPS/InferReal 
```
- View Synthesis from a Virtual Viewpoint:
```
python inference/infer_virtual.py \
    --ckpt experiments/NovelViewSynthesis/Panoptic/256x256/LPIPS/model.ckpt \
    --root /path/to/datasets/Panoptic \
    --sequence 171026_pose1/Subsequences/0 \
    --view 00 \
    --interval 2922 3751 \
    --output_dir experiments/NovelViewSynthesis/Panoptic/256x256/LPIPS/InferVirtual
```
- Motion Transfer between Actors:
```
python inference/motion_transfer.py \
    --ckpt experiments/NovelViewSynthesis/Panoptic/256x256/LPIPS/model.ckpt \
    --root /path/to/datasets/Panoptic \
    --sequence_motion 171204_pose1/Subsequences/0 \
    --in_view_motion 00 \
    --out_view_motion 24 \
    --interval_motion 3903 4237 \
    --sequence_appearance 171026_pose1/Subsequences/1 \
    --in_view_appearance 18 \
    --interval_appearance 5445 5445 \
    --output_dir experiments/NovelViewSynthesis/Panoptic/256x256/LPIPS/MotionTransfer
```

## Pre-trained Models
Download the following models and place them in their correspoding directories:
- [Human36M.256x256.LPIPS.ckpt](https://github.com/GuillaumeRochette/HumanViewSynthesis/releases/download/v0.0.1-alpha/Human36M.256x256.LPIPS.ckpt) in
`experiments/NovelViewSynthesis/Human36M/256x256/LPIPS`.
- [Panoptic.512x512.LPIPS.ckpt](https://github.com/GuillaumeRochette/HumanViewSynthesis/releases/download/v0.0.1-alpha/Panoptic.512x512.LPIPS.ckpt) in `experiments/NovelViewSynthesis/Panoptic/512x512/LPIPS`.

## Visualise the Renderer's Output

- Simple shapes (one at a time):
```
python rendering/visualize_shapes.py \
    -- one_sphere \
#    --one_ellipsoid \
#    --two_spheres \
#    --many_spheres 64 \
#    --many_ellipsoids 64 \
    --height 270 \
    --width 480
```
- Human Poses:
```
python rendering/visualize_poses.py \
    --root /path/to/datasets/Panoptic \
    --sequence 171026_pose1/Subsequences/0 \
    --view 00 \
    --frame 420 \
    --height 256 \
    --width 256
```

## Citation
```
@article{Rochette2021,
  title={Human Pose Manipulation and Novel View Synthesis using Differentiable Rendering},
  author={Rochette, Guillaume and Russell, Chris and Bowden, Richard},
  booktitle={IEEE International Conference on Automatic Face and Gesture Recognition (FG), 2021},
  year={2021}
}
```
