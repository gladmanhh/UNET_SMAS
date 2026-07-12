# PicoSAM2BaseUNet

최종 full-frame, prompt-free segmentation 구현입니다. 전체 사용법은 저장소
루트의 [`README.md`](../README.md)를 먼저 참고하세요.

## Final path

```text
RGB frame (320 x 192)
  -> depthwise separable U-Net
  -> background / dermis / SMAS / bone
  -> postprocessing
  -> five-layer MP4 visualization
```

기본 checkpoint:

```text
checkpoints/picosam2_unet_320x192.pt
```

## Inference

저장소 root에서:

```bash
python PICOSAM2baseUNet/launcher.py infer-bone 25
```

이 명령은 아래 입력을 자동으로 찾습니다.

```text
data/frames/output25/frames/*.png
```

명시적인 경로도 사용할 수 있습니다.

```bash
python PICOSAM2baseUNet/infer_bone_multiclass.py custom \
  --frames-dir /path/to/frames
```

최종 결과:

```text
PICOSAM2baseUNet/outputs_bone_multiclass/output25/output25_segmentation.mp4
```

추론 pipeline depth 비교:

```bash
python PICOSAM2baseUNet/launcher.py infer-bone 25 --pipeline-depth 1
python PICOSAM2baseUNet/launcher.py infer-bone 25 --pipeline-depth 2
python PICOSAM2baseUNet/launcher.py infer-bone 25 --pipeline-depth 3
```

## Training

```bash
python PICOSAM2baseUNet/train_bone_multiclass.py \
  --annotations data/annotations.json \
  --dermis-root data/dermis \
  --bone-root data/bone \
  --width 320 --height 192 \
  --epochs 8 --batch-size 12
```

데이터 구조는 [`data/README.md`](../data/README.md)에 정리되어 있습니다.

## Files

| File | Role |
| --- | --- |
| `model.py` | depthwise separable U-Net architecture |
| `infer_bone_multiclass.py` | final inference, postprocessing, pipeline, video output |
| `train_bone_multiclass.py` | final 4-class supervised training |
| `launcher.py` | short train/inference commands |
| `common.py` | data discovery and image conversion helpers |
| `checkpoints/model_card.json` | checkpoint metadata and validation metrics |

`infer.py`, `train.py`, `infer_multiclass.py`, and `train_multiclass.py` are kept
for the earlier SMAS-only and dermis+SMAS stages.
