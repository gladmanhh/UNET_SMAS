# PicoSAM2BaseUNet

최종 full-frame, prompt-free 4-class segmentation 구현입니다. 모델은 한 장의 RGB
frame을 입력받아 `background`, `dermis`, `SMAS`, `bone`을 예측합니다. 렌더링할
때 경계 사이를 채워 `dermis`, `subc`, `SMAS`, `muscle`, `bone`의 5개 층으로
표시합니다.

```text
RGB frame (320 x 192)
  -> depthwise separable U-Net
  -> background / dermis / SMAS / bone
  -> boundary postprocessing
  -> five-layer MP4 visualization
```

## Inference

저장소 root에서 다음 명령을 실행합니다.

```bash
python PICOSAM2baseUNet/infer.py 25
```

입력과 출력 경로는 다음과 같습니다.

```text
input:  data/frames/output25/frames/*.png
output: PICOSAM2baseUNet/outputs/output25/output25_segmentation.mp4
```

임의의 frame 폴더를 지정할 수도 있습니다.

```bash
python PICOSAM2baseUNet/infer.py custom \
  --frames-dir /path/to/frames
```

기본 checkpoint는 `checkpoints/picosam2_unet_320x192.pt`이며 모든 frame을
처리한 뒤 최종 MP4 하나만 저장합니다.

## Training

```bash
python PICOSAM2baseUNet/train.py
```

`data/images/<split>`과 `data/masks/<split>` 아래에서 이름이 같은 image/mask를
자동으로 짝지으므로 annotation 파일이나 외부 label 경로가 필요하지 않습니다.
mask는 `0=background`, `1=dermis`, `2=SMAS`, `3=bone`인 class-index PNG입니다.

기본 입력 크기는 `320 x 192`, epoch은 8, batch size는 12입니다. 기본 split은
output 1, 20, 30, 42를 train에 사용하고 output 10을 validation에 사용하며,
output 42는 frame 290까지 포함합니다. 데이터 구조는
[`data/README.md`](../data/README.md)를 참고하세요.

## Files

| File | Role |
| --- | --- |
| `model.py` | depthwise separable U-Net architecture |
| `train.py` | final 4-class supervised training |
| `infer.py` | inference, postprocessing, rendering, MP4 output |
| `common.py` | data discovery and image conversion helpers |
| `checkpoints/model_card.json` | checkpoint metadata and validation metrics |
