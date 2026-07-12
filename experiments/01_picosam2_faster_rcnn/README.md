# 01_picosam2_faster_rcnn

Experiment: Faster R-CNN bbox detection + crop PicoSAM2-style U-Net segmentation.

This folder is kept as an experiment only. The main inference path is now:

```text
PICOSAM2baseUNet
```

Train the crop segmentation model for this experiment:

```powershell
python .\experiments\01_picosam2_faster_rcnn\train.py
```

Generated checkpoints and visual outputs are intentionally excluded from Git.
