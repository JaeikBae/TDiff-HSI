![Graphical Abstract](img/Graphical%20Abstract_new.png)

🌐 [En](#en) | [Ko](#ko)
---

## 📄 Paper

**TDiff-HSI: Tucker-guided diffusion for high-dimensional RGB-to-HSI image generation**  
Jaeik Bae, Yong-Gu Lee  
*Journal of Computational Design and Engineering*, 13(2), 197–212, 2026  

🔗 https://doi.org/10.1093/jcde/qwag008

---
<a id="en"></a>
## Model and Dataset Release 🚀

🧭 Contents

- [Assets Summary](#en-assets)
- [Download](#en-download)
- [Docker](#en-docker)
- [Quickstart: Inference](#en-inference)
- [Reproducing Training](#en-train)
- [Paper](#en-paper)
- [Contact](#en-contact)

This repository releases both the trained model and the datasets used for training/evaluation. Datasets are provided in two archive formats: `tar` and `zip`. Trained model weights are provided as `.pth` files (two parts: diffusion and upsample).

You can selectively download only what you need. Large files are distributed via Google Drive links.

<a id="en-assets"></a>
### 📦 Assets Summary

Pick one of the two dataset archives (tar/zip). Content is identical.

| Item | Description | Format | Link |
|---|---|---|---|
| Dataset (tar) | Train/Val/Test data | `.tar` | [Dataset (tar)](https://drive.google.com/file/d/1LuJwGNK6Mrk7TyBFuQk-311IQa_TweOR/view?usp=sharing) |
| Dataset (zip) | Train/Val/Test data | `.zip` | [Dataset (zip)](https://drive.google.com/file/d/1z8WR81dqdwKS4E5PIkzxoB1a1HVTHijS/view?usp=sharing) |
| Diffusion model weights | Trained diffusion model | `.pth` | [Download](https://drive.google.com/file/d/10dV23EDZmOAytgbXDNPwZ6essYvBriF3/view?usp=sharing) |
| Upsample model weights | Trained upsample model | `.pth` | [Download](https://drive.google.com/file/d/1dEKVPADcDYf1Q5sCw4vqarhSla18pzxB/view?usp=sharing) |

<a id="en-download"></a>
### ⬇️ Download

- Manual: Open the Google Drive links above in your browser.

<a id="en-extract"></a>
### 📂 Extract Archives

```bash
# tar archives (dataset only)
tar -xvf dataset.tar -C /desired/path

# zip archives (dataset only)
unzip dataset.zip -d /desired/path
```

<a id="en-structure"></a>
### 🗂️ Dataset Directory Structure (Example)

```text
dataset_root/
  train/
    ...
  val/
    ...
  test/
    ...
```

The exact sub-structure and file formats may vary by project. Please update this README to reflect the actual released dataset structure.

<a id="en-env"></a>
### 🛠️ Environment Setup

```bash
python -V                 # Check Python version (recommended: 3.9+)
pip install -r requirements.txt
```

Tips

- Use GPU if available for both training and inference.
- Default paths assume running inside the container with `/app` as project root.

<a id="en-docker"></a>
### 🐳 Docker

Build and run with Docker (requires Docker installed):

```bash
# Build image
docker build -t diffusion-hsi-img -f Dockerfile .

# Run container (GPU + larger shared memory recommended)
docker run --rm -it \
  --gpus all \
  --shm-size=64G \
  -v $(pwd):/app \
  diffusion-hsi-img bash

# (Optional) using the provided helper
bash docker_build_run.sh
```

<a id="en-inference"></a>
### ⚡ Quickstart: Inference

Run the evaluation/inference script:

```bash
python models/test.py \
  --data_dir /app/datas/hsi \
  --model /app/weights/diffusion_model.pth \
  --upsample_model /app/weights/upsample_model.pth \
  --save_base /app/results
```

<a id="en-train"></a>
### 🔁 Reproducing Training

Option A (recommended helper):

```bash
cd models
bash run_train_diffusion.sh 4 \
  --data_dir /app/datas/hsi \
  --data_dir_test /app/datas/val \
  --save_dir /app/weights \
  --batch_size 2 \
  --epochs 5000 \
  --num_workers 4
```

Option B (direct):

```bash
OMP_NUM_THREADS=16 \
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  models/train_diffusion.py \
  --data_dir /app/datas/hsi \
  --data_dir_test /app/datas/val \
  --save_dir /app/weights \
  --batch_size 2 \
  --epochs 5000 \
  --num_workers 4
```

Checkpoints will be saved under `/app/weights` at the configured intervals.

<a id="en-paper"></a>
### 📄 Paper

**TDiff-HSI: Tucker-guided diffusion for high-dimensional RGB-to-HSI image generation**  
Jaeik Bae, Yong-Gu Lee  
*Journal of Computational Design and Engineering*, 13(2), 197–212, 2026  

🔗 https://doi.org/10.1093/jcde/qwag008

<a id="en-contact"></a>
### ✉️ Contact

Please open a GitHub Issue or reach out via:

- Email: jaeikb38@gm.gist.ac.kr

---

ℹ️ Note: The following section is in Korean.
<a id="ko"></a>

## 모델 및 데이터 공개 안내 🚀

🌐 [En](#en) | [Ko](#ko)

🧭 목차

- [구성 요약](#ko-assets)
- [다운로드 방법](#ko-download)
- [압축 해제](#ko-extract)
- [데이터셋 디렉터리 구조](#ko-structure)
- [환경 준비](#ko-env)
- [Docker](#ko-docker)
- [빠른 시작: 추론(Inference)](#ko-inference)
- [학습(Training) 재현](#ko-train)
- [논문](#ko-paper)
- [문의](#ko-contact)

이 저장소는 학습된 모델과 학습/평가에 사용된 데이터셋을 함께 공개합니다. 데이터셋은 `tar`와 `zip` 두 가지 압축 형식으로 제공되며, 학습된 모델 가중치는 `.pth` 파일(디퓨전/업샘플 2파트)로 제공합니다.

필요한 항목만 골라 받으실 수 있으며, 대용량 파일은 Google Drive 링크를 통해 다운로드하실 수 있습니다.

<a id="ko-assets"></a>
### 📦 구성 요약

| 항목 | 설명 | 형식 | 링크 |
|---|---|---|---|
| 데이터셋 (tar) | 학습/검증/테스트 데이터 | `.tar` | [데이터셋 (tar)](https://drive.google.com/file/d/1LuJwGNK6Mrk7TyBFuQk-311IQa_TweOR/view?usp=sharing) |
| 데이터셋 (zip) | 학습/검증/테스트 데이터 | `.zip` | [데이터셋 (zip)](https://drive.google.com/file/d/1z8WR81dqdwKS4E5PIkzxoB1a1HVTHijS/view?usp=sharing) |
| 디퓨전 모델 가중치 | 학습된 디퓨전 모델 | `.pth` | [다운로드](https://drive.google.com/file/d/1S7MCVovFixrIuOLl7oVWRdlMd5kJwG6r/view?usp=sharing) |
| 업샘플 모델 가중치 | 학습된 업샘플 모델 | `.pth` | [다운로드](https://drive.google.com/file/d/1EyPDn008j1MA-w36OgIwt93xBR8Lo6YB/view?usp=sharing) |

<a id="ko-download"></a>
### ⬇️ 다운로드 방법

- 수동 다운로드: 위 표의 Google Drive 링크를 브라우저에서 열어 다운로드합니다.
- CLI 다운로드: `gdown`을 사용하면 대용량 파일을 커맨드라인에서 받을 수 있습니다.

```bash
# gdown 설치
pip install gdown

# 데이터셋 (tar)
gdown --fuzzy "https://drive.google.com/file/d/1LuJwGNK6Mrk7TyBFuQk-311IQa_TweOR/view?usp=sharing"

# 데이터셋 (zip)
gdown --fuzzy "https://drive.google.com/file/d/1z8WR81dqdwKS4E5PIkzxoB1a1HVTHijS/view?usp=sharing"

# 디퓨전 모델 가중치
gdown --fuzzy "https://drive.google.com/file/d/1S7MCVovFixrIuOLl7oVWRdlMd5kJwG6r/view?usp=sharing"

# 업샘플 모델 가중치
gdown --fuzzy "https://drive.google.com/file/d/1EyPDn008j1MA-w36OgIwt93xBR8Lo6YB/view?usp=sharing"
```

 

<a id="ko-extract"></a>
### 📂 압축 해제

```bash
# tar 형식 (데이터셋만 해당)
tar -xvf 데이터셋.tar -C 원하는_경로

# zip 형식 (데이터셋만 해당)
unzip 데이터셋.zip -d 원하는_경로
```

<a id="ko-structure"></a>
### 🗂️ 데이터셋 디렉터리 구조 (예시)

```text
dataset_root/
  train/
    ...
  val/
    ...
  test/
    ...
```

프로젝트에 따라 하위 구조와 파일 포맷은 다를 수 있습니다. 실제 공개되는 데이터 구조를 간단히 README에 보완해 주세요.

<a id="ko-env"></a>
### 🛠️ 환경 준비

```bash
python -V                 # Python 버전 확인 (권장: 3.9+)
pip install -r requirements.txt
```

<a id="ko-docker"></a>
### 🐳 Docker

Docker로 빌드 및 실행 (Docker 설치 필요):

```bash
# 이미지 빌드
docker build -t diffusion-hsi-img -f Dockerfile .

# 컨테이너 실행 (GPU + 충분한 shared memory 권장)
docker run --rm -it \
  --gpus all \
  --shm-size=64G \
  -v $(pwd):/app \
  diffusion-hsi-img bash

# (선택) 제공된 스크립트 사용
bash docker_build_run.sh
```

<a id="ko-inference"></a>
### ⚡ 빠른 시작: 추론(Inference)

평가/추론 스크립트를 실행합니다:

```bash
python models/test.py \
  --data_dir /app/datas/hsi \
  --model /app/weights/diffusion_model.pth \
  --upsample_model /app/weights/upsample_model.pth \
  --save_base /app/results
```

<a id="ko-train"></a>
### 🔁 학습(Training) 재현

방법 A (권장, 헬퍼 스크립트):

```bash
cd models
bash run_train_diffusion.sh 4 \
  --data_dir /app/datas/hsi \
  --data_dir_test /app/datas/val \
  --save_dir /app/weights \
  --batch_size 2 \
  --epochs 5000 \
  --num_workers 4
```

방법 B (직접 실행):

```bash
OMP_NUM_THREADS=16 \
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  models/train_diffusion.py \
  --data_dir /app/datas/hsi \
  --data_dir_test /app/datas/val \
  --save_dir /app/weights \
  --batch_size 2 \
  --epochs 5000 \
  --num_workers 4
```

체크포인트는 설정된 주기에 `/app/weights` 아래에 저장됩니다.

<a id="ko-paper"></a>
### 📄 논문

**TDiff-HSI: Tucker-guided diffusion for high-dimensional RGB-to-HSI image generation**  
Jaeik Bae, Yong-Gu Lee  
Journal of Computational Design and Engineering, 13(2), 197–212, 2026  

🔗 https://doi.org/10.1093/jcde/qwag008

 

<a id="ko-contact"></a>
### ✉️ 문의

이슈나 질문은 GitHub Issues를 통해 남겨주시거나 아래 연락처로 문의해 주세요.

- 이메일: jaeikb38@gm.gist.ac.kr



 

