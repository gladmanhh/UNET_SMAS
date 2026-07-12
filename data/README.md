# Data layout

Ultrasound images and masks are not included in this repository. Place local
data under this directory before training or inference.

## Inference frames

```text
data/
  frames/
    output25/
      frames/
        frame_00001.png
        frame_00002.png
        ...
```

The inference command accepts `25` or `output25` and discovers this directory
automatically. An explicit directory can also be passed with `--frames-dir`.

## Training labels

The final multiclass trainer expects the following logical layout:

```text
data/
  annotations.json
  frames/
    output1/frames/*.png
  masks/
    ... SMAS masks referenced by annotations.json ...
  dermis/
    output1/dermis_masks/*.png
  bone/
    output1/labels/*.png
```

Each `annotations.json` item must contain at least:

```json
{
  "id": "output1_frame_00001",
  "output": "output1",
  "frame": "frame_00001.png",
  "image_path": "data/frames/output1/frames/frame_00001.png",
  "mask_path": "data/masks/train/output1_frame_00001.png"
}
```

`image_path` and `mask_path` are resolved relative to the repository root.
Dermis and bone roots can be changed with `--dermis-root` and `--bone-root`.

The default final-model split is:

```text
train: output1, output20, output30, output42 (output42 up to frame 290)
val:   output10
```
