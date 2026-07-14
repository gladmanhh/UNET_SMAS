# Dataset layout

The training dataset is distributed separately from the GitHub repository.
After downloading it, place the `images` and `masks` folders directly under
`data`:

```text
data/
  images/
    train/
      output1_frame_00001.png
      ...
    val/
      output10_frame_00001.png
      ...
    test/                 # optional inference/evaluation images
      output44_frame_00001.png
      ...
  masks/
    manifest.json
    train/
      output1_frame_00001.png
      ...
    val/
      output10_frame_00001.png
      ...
```

Every training image and mask has exactly the same relative filename. The
trainer discovers these pairs directly, so `annotations.json` and machine-
specific dermis/bone paths are not required.

## Mask format

Each mask is a single-channel paletted PNG whose pixel value is the class index:

| Pixel value | Class | Display color |
| ---: | --- | --- |
| 0 | background | black |
| 1 | dermis | cyan |
| 2 | SMAS | yellow |
| 3 | bone | magenta |

The colors are only the PNG palette used for convenient viewing. Training uses
the integer pixel values `0` through `3`. Where source binary masks overlapped,
the conversion priority was `bone > SMAS > dermis`.

## Default split

```text
train: output1, output20, output30, output42 (539 fully labeled frames)
val:   output10                              (72 fully labeled frames)
```

Only frames with complete dermis, SMAS, and bone labels are present in the
training masks. `output42` is limited to frame 290; its frame 290 bone label was
not available, so the packaged output42 portion contains frames 1 through 289.

`output44` has no bone ground truth. Its images may be included under
`images/test` for inference, but it is intentionally not packaged as a complete
4-class ground-truth mask set.

## Train

From the repository root:

```bash
python PICOSAM2baseUNet/train.py
```

Useful overrides:

```bash
python PICOSAM2baseUNet/train.py \
  --data-root data \
  --width 320 --height 192 \
  --epochs 8 --batch-size 12
```

Inference frames use a separate layout:

```text
data/frames/output25/frames/frame_00001.png
```
