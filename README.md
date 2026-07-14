# UNET SMAS

B-mode ultrasound frame에서 해부학적 층을 분할하고 영상으로 저장하는
경량 U-Net 프로젝트입니다. 최종 모델은 bbox나 point prompt 없이 전체 frame
이미지만 입력받습니다.

```text
ultrasound frame
  -> PicoSAM2-style depthwise separable U-Net
  -> background / dermis / SMAS / bone
  -> boundary postprocessing
  -> dermis / subc / SMAS / muscle / bone visualization
  -> MP4 video
```

모델이 직접 학습하는 class는 `background`, `dermis`, `SMAS`, `bone`입니다.
`subc`는 dermis와 SMAS 사이, `muscle`은 SMAS와 bone 사이 영역으로 구성하여
최종 영상에는 총 5개 층을 표시합니다.

## Model profile

| Item | Value |
| --- | ---: |
| Input | 320 x 192 RGB |
| Parameters | 914,885 |
| Checkpoint | 3.60 MiB |
| Profiled compute | 3.28 GFLOPs/frame |
| Validation mean IoU | 0.8935 |
| Validation SMAS IoU | 0.7753 |

Checkpoint와 세부 metric은
[`PICOSAM2baseUNet/checkpoints`](PICOSAM2baseUNet/checkpoints)에 포함되어
있습니다.

## Repository structure

```text
UNET_SMAS/
  PICOSAM2baseUNet/        # final 4-class model, training, inference
    checkpoints/           # public final checkpoint and model card
  experiments/             # archived detector/crop alternatives (source only)
  data/README.md            # private data layout
  requirements.txt
```

원본 ultrasound frame, mask, 생성 영상, 중간 checkpoint는 저장소에 포함하지
않습니다.

## Installation

### Windows

```powershell
git clone https://github.com/leejinwoo56/UNET_SMAS.git
cd UNET_SMAS
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
```

CUDA를 사용할 경우 PyTorch는 로컬 driver에 맞는 build로 먼저 설치하세요.

### Linux / Raspberry Pi

PyTorch와 OpenCV는 사용 중인 Raspberry Pi OS와 Python 버전에 맞는 CPU build를
먼저 설치합니다. 이후 저장소를 clone하고 동일한 환경에서 아래 import를
확인합니다.

```bash
git clone https://github.com/leejinwoo56/UNET_SMAS.git
cd UNET_SMAS
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
python -c "import torch, cv2; print(torch.__version__, cv2.__version__)"
```

## Quick start: inference

아래 경로는 학습 데이터 경로가 아니라 `output` 번호로 실행하는 추론용 frame
경로입니다. 먼저 frame을 다음과 같이 배치합니다.

```text
data/frames/output25/frames/frame_00001.png
data/frames/output25/frames/frame_00002.png
...
```

프로젝트 root에서 실행합니다. `25`를 입력하면 `output25` 전체 frame을 자동으로
읽습니다.

```bash
python PICOSAM2baseUNet/infer.py 25
```

결과는 하나의 영상으로 저장됩니다.

```text
PICOSAM2baseUNet/outputs/output25/output25_segmentation.mp4
```

다른 frame 폴더를 직접 지정하려면:

```bash
python PICOSAM2baseUNet/infer.py custom \
  --frames-dir /path/to/frames
```

## Inference pipeline

추론은 다음 세 단계를 겹쳐 실행합니다.

```text
frame loading + resize | model inference | postprocess + render + video write
```

`pipeline-depth`는 동시에 준비하거나 대기할 수 있는 frame 수이며 batch size나
U-Net depth가 아닙니다. CPU 환경에서는 `1`, `2`, `3`을 비교하여 total FPS가
가장 높은 값을 사용하세요.

```bash
python PICOSAM2baseUNet/infer.py 25 --pipeline-depth 1
python PICOSAM2baseUNet/infer.py 25 --pipeline-depth 2
python PICOSAM2baseUNet/infer.py 25 --pipeline-depth 3
```

후처리 없이 raw model 경향을 확인할 수도 있습니다.

```bash
python PICOSAM2baseUNet/infer.py 25 --no-connect-smas-edges
python PICOSAM2baseUNet/infer.py 25 --no-clean-bone
```

## Training

학습 데이터는 [Google Drive에서 다운로드](https://drive.google.com/drive/folders/1uNTCNoBQWgii9hYUwY3jDkA3U5c1iJ4F)한 뒤,
Drive의 `images`, `masks` 폴더가 각각 `data/images`, `data/masks`가 되도록
배치합니다. 상세 구조는 [`data/README.md`](data/README.md)를 참고하세요.

```bash
python PICOSAM2baseUNet/train.py
```

trainer는 `data/images`와 `data/masks`에서 같은 이름의 파일을 자동으로
짝지어 읽습니다. mask 한 장에 `background=0`, `dermis=1`, `SMAS=2`,
`bone=3` class index가 저장되어 있어 별도의 외부 label 경로가 필요하지 않습니다.

기본 split은 `output1, output20, output30, output42`를 train에 사용하고
`output10`을 validation에 사용합니다. `output42`는 frame 290까지만 train에
포함합니다.

## Postprocessing

- Bone: 작은/상단 component를 제거하고 가장 큰 하단 component의 경계 아래를 채웁니다.
- SMAS: 작은 분리 component를 제거하고, 주 component가 전체 면적의 1.5%보다
  작으면 SMAS가 없는 frame으로 처리합니다.
- SMAS boundary: 양쪽 edge 연결, 최대 기울기 제한, 급격한 V-shaped spike 완화를
  적용합니다.
- Render: 저해상도에서 후처리한 뒤 OpenCV LUT 합성과 MP4 저장을 수행합니다.

## Benchmarks

`output25` 92 frames, render size 1050 x 630 기준 내부 측정값입니다. 환경에 따라
달라질 수 있습니다.

| Environment | Model FPS | End-to-end FPS | Pipeline depth |
| --- | ---: | ---: | ---: |
| RTX 4090 desktop | about 70 | about 35 | 3 |
| Raspberry Pi 5 / CM5, CPU | 1.91 | 1.91 | 2 |

Raspberry Pi 측정에서는 model inference가 전체 병목이었습니다. ONNX Runtime,
INT8 static quantization, CPU thread tuning은 후속 최적화 대상으로 남아 있습니다.

## Experiments

`experiments/`에는 최종 경로로 채택하지 않은 조합의 source code만 남겨두었습니다.

- `01_picosam2_faster_rcnn`: Faster R-CNN bbox + crop segmentation
- `02_picosam2_tinybbox_detect`: Tiny bbox detector + crop segmentation
- `03_faster_rcnn_analog`: Faster R-CNN bbox + brightness-based layer extraction

최종 사용 경로는 `PICOSAM2baseUNet`입니다.
