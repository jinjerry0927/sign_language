# 한국수어 단어 인식기 (KSL Word Recognizer)

MediaPipe로 손·상체 랜드마크를 추출하고 LSTM으로 한국수어 단어를 분류하는 MVP 프로젝트.

> Repo: https://github.com/jinjerry0927/sign_language

## 인식 가능한 단어 (10개)

엄마 · 누나 · 형 · 밥 · 노래 · 꿈 · 자다 · 놀다 · 모르다 · 없다

## 데모

```powershell
# 영상 파일로 예측
python src\predict.py --video test_videos\kkum_dream.mp4

# 웹캠 실시간 예측
python src\predict.py
```

좌측 상단 HUD가 손/포즈 검출 상태를, 큰 글씨로 예측 단어 + 신뢰도를 표시합니다.

## 기술 스택

- Python 3.11
- MediaPipe Tasks API — `HolisticLandmarker` (양손 21점 + 상체 33점, 총 75 × 3D)
- TensorFlow / Keras — LSTM(64) 분류기
- OpenCV (영상) · PIL (한국어 자막)

학습 데이터는 [AI Hub 한국수어 영상 데이터셋](https://aihub.or.kr) (`004.수어영상`)의 화자 REAL01 정면(F) 영상.

## 설치 + 실행

```powershell
# 1. 가상환경
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. 설치 검증
python src\verify_install.py

# 3. (선택) AI Hub zip을 가지고 있다면 테스트용 영상 10개 추출
#    -> data\raw\01_real_word_video.zip 위치에 두고 실행
python tools\extract_test_videos.py

# 4. 실행
python src\predict.py                          # 웹캠
python src\predict.py --video <ascii_path.mp4> # 영상 파일
```

> **Windows + 한글 경로 주의**: cv2.VideoCapture가 한글 경로 영상을 못 엽니다. `predict.py`는 자동으로 ASCII 임시 경로로 복사하지만, 가급적 영문 폴더에 두는 편이 안정적입니다.

## 프로젝트 구조

```
sign_language/
├── src/                          # 핵심 파이프라인
│   ├── landmarks.py              # MediaPipe 랜드마크 추출 + 시각화
│   ├── dataset.py                # 정규화 + 시퀀스 길이 통일 + 증강
│   ├── model.py                  # LSTM 정의
│   ├── train.py                  # 학습 루프
│   ├── predict.py                # 실시간 추론 (웹캠/영상)
│   ├── extract_dataset.py        # AI Hub 영상 zip → 랜드마크 .npy
│   ├── webcam_demo.py            # MediaPipe 시각화 데모 (학습 모델 없이 동작)
│   ├── verify_install.py         # 설치 검증
│   └── eval_on_npy.py            # 학습된 모델 성능 점검
├── tools/                        # 보조 유틸리티
│   ├── build_word_mapping.py     # AI Hub morpheme zip → word_mapping.csv
│   ├── join_parts.py             # AI Hub 분할 다운로드(.partN) 재결합
│   ├── probe_camera.py           # 웹캠 인덱스/백엔드 진단
│   ├── extract_test_videos.py    # zip → test_videos/ ASCII 영상 10개
│   └── inspect_landmarks.py      # 추출 .npy 통계
├── data/
│   ├── target_words.csv          # 인식 단어 10개 (word_id ↔ 한국어)
│   └── word_mapping.csv          # AI Hub 전체 3000 단어 매핑
├── models/
│   └── holistic_landmarker.task  # MediaPipe (14 MB)
├── checkpoints/
│   └── best.keras                # 학습된 LSTM (300 KB)
├── requirements.txt
└── README.md
```

`.gitignore`로 제외된 것: `venv/`, `data/raw/*.zip` (AI Hub 원본 ~32 GB), `data/landmarks/*.npy`, `test_videos/*.mp4`, 런타임 산출물.

## 재학습 흐름

1. AI Hub에서 `004.수어영상` Training `01_real_word_video.zip` 다운로드 → `data/raw/`에 배치 (필요 시 `tools/join_parts.py`로 분할 파일 재결합)
2. `python src/extract_dataset.py --zip data/raw/01_real_word_video.zip --angle F` → `data/landmarks/*.npy`
3. `python src/train.py` → `checkpoints/best.keras` 갱신
4. `python src/predict.py` 로 테스트

단어를 바꾸려면 `data/target_words.csv`를 수정하고 1~3단계 재실행.

## 한계

- **단일 화자 학습**: 화자 REAL01 1명 × 단어당 영상 1개로 학습. 다른 사람 웹캠에는 일반화가 떨어질 수 있음.
- **검증 정확도 100%의 함정**: 검증셋은 학습 anchor의 다른 augmentation 변형이라 진짜 일반화 측정값이 아님.
- **F각도 전용**: 정면 영상만 학습 — 측면/위아래 시점은 인식 어려움.

## 개선 아이디어

- 다른 화자 zip 추가 다운 (`02_real_word_video.zip` 등) → 다중 화자 학습
- D/L/R/U 각도 영상도 학습에 포함 → 시점 견고성
- 직접 녹화 영상 anchor에 추가 → 본인 화자 적응
- Transformer/3D CNN 등 모델 확장
- 단어 수 50개 이상으로 확장

## 라이선스 / 데이터 출처

- 코드: 학습용 개인 프로젝트 (별도 라이선스 명시 없음)
- 학습 영상: AI Hub "한국수어 영상" 데이터셋 — AI Hub 이용약관을 따름
- 영상 원본은 저장소에 포함하지 않음 (`.gitignore`로 제외)
