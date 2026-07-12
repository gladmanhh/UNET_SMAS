# 03_faster_rcnn_analog

Experiment: Faster R-CNN bbox detection + brightness/analog SMAS band extraction.

This folder is a lightweight detection experiment derived from
`CLASSYS-BEAUTY/data`.

Goal:

- Convert SMAS binary segmentation masks into one bbox per frame.
- Train a CNN-based bbox detector as a first-stage SMAS region proposal model.
- Evaluate with output-level train/val/test splits to avoid frame leakage.

Initial split:

```text
train: output1, output20, output30, output42
val:   output10
test:  output44
```

Only labeled outputs inside `output1` to `output44` are used. In the current
data, those are `output1`, `output10`, `output20`, `output30`, `output42`,
and `output44`.

The first model is `torchvision` Faster R-CNN with a MobileNetV3 320 FPN
backbone. It is still a Faster R-CNN style detector, but the backbone and
input scale are intentionally lightweight compared with MedSAM2.

## Run

Activate the repository virtual environment first.

Easy launcher:

```powershell
python .\launcher.py
python .\launcher.py train --epochs 12 --batch-size 4 --run-name mobilenetv3_retrain
python .\launcher.py infer 1
```

CUDA is selected automatically when it is available; otherwise the scripts fall
back to CPU.

Prepare the dataset:

```powershell
cd .\experiments\03_faster_rcnn_analog
python .\prepare_bbox_dataset.py
```

Train and evaluate:

```powershell
python .\train_detector.py --epochs 12 --batch-size 4
```

Train the detector with dataset preparation in one step:

```powershell
python .\train.py
```

Run inference on `CLASSYS-BEAUTY\data\frames\output1\frames`:

```powershell
python .\infer_detector.py
# enter 1 at the prompt
```

Or run it without the prompt:

```powershell
python .\infer_detector.py 1
```

Detected frames are saved as:

```text
outputs/output1/frames/frame__00001_detect.png
```

Important outputs:

```text
data/annotations.json
runs/<run_name>/best.pt
runs/mobilenetv3_frcnn_8ep/best.pt
outputs/<output_name>/frames/*_detect.png
runs/<run_name>/final_metrics.json
runs/<run_name>/predictions_test.csv
runs/<run_name>/visualizations/test/
```

Metrics are bbox metrics, not segmentation metrics:

- AP50 and mAP50:95
- top-1 bbox IoU
- precision/recall at score >= 0.5 and IoU >= 0.5
- false positives on empty-mask frames
- single-image inference FPS
