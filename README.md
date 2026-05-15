# 한국수어 단어 인식기 (KSL Word Recognizer)

MediaPipe로 손/상체 랜드마크를 추출하고 LSTM으로 한국수어 단어를 분류하는 MVP 프로젝트.

## 인식 가능한 단어 (10개)

엄마 · 누나 · 형 · 밥 · 노래 · 꿈 · 자다 · 놀다 · 모르다 · 없다

## 기술 스택

- **Python 3.11**
- **MediaPipe Tasks API** — `HolisticLandmarker` (양손 각 21점 + 상체 포즈 33점)
- **TensorFlow / Keras** — LSTM 분류기 (64 unit)
- **OpenCV** — 영상 입출력
- **PIL** — 한국어 자막 렌더링

학습 데이터는 [AI Hub 한국수어 영상 데이터셋](https://aihub.or.kr) (`004.수어영상`)에서 화자 1명(REAL01)의 정면(F각도) 영상만 사용.

## 실행

```powershell
# 가상환경
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# MediaPipe HolisticLandmarker 모델 다운로드 (14 MB)
# 원하면 깃 추적할 수도, 직접 받을 수도 있음. 기본은 추적함.
# 직접 받으려면:
#   Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task" -OutFile "models/holistic_landmarker.task"

# 설치 확인
python src/verify_install.py

# 실시간 웹캠 예측
python src/predict.py

# 영상 파일로 예측
python src/predict.py --video test_videos/kkum_dream.mp4

# 모델 재학습 (data/landmarks/*.npy 가 있어야 함)
python src/train.py
```

## 프로젝트 구조

```
sign_language/
├── src/
│   ├── landmarks.py            # MediaPipe 추출 모듈
│   ├── dataset.py              # 정규화 + 증강
│   ├── model.py                # LSTM 정의
│   ├── train.py                # 학습 루프
│   ├── predict.py              # 실시간 추론 (웹캠/영상)
│   ├── extract_dataset.py      # AI Hub 영상 zip → .npy
│   ├── webcam_demo.py          # Phase 1 시각화 데모
│   ├── verify_install.py       # 설치 검증
│   ├── eval_on_npy.py          # 학습 결과 점검
│   └── ...                     # 부속 도구들
├── data/
│   ├── target_words.csv        # 인식 단어 10개 (word_id ↔ 한국어)
│   └── word_mapping.csv        # AI Hub 전체 3000 단어 매핑
├── models/
│   └── holistic_landmarker.task
├── checkpoints/
│   └── best.keras              # 학습된 LSTM
└── requirements.txt
```

## 한계 (솔직한 기록)

- **단일 화자 학습**: AI Hub Training zip 01의 화자 REAL01 1명 데이터만 사용. 다른 사람 웹캠 영상에는 일반화가 떨어질 수 있음.
- **단어당 영상 1개**: 데이터 증강(좌우반전·노이즈·시간왜곡)으로 변형은 만들지만 본질적으로 1개 동작의 변형.
- **검증 정확도 100%의 함정**: 검증셋은 학습 anchor의 다른 증강 변형이라 실제 일반화 측정값 아님.

## 개선 방향

- 다른 화자 zip 추가 다운로드 (`02_real_word_video.zip` 등) → 다중 화자 학습
- D/L/R/U 각도 영상도 학습에 포함 → 시점 견고성
- 직접 녹화한 영상 anchor에 추가 → 본인 화자에 적응
- Transformer/3D CNN 등 모델 확장

## 데이터 라이선스

학습에 쓰는 영상은 AI Hub "한국수어 영상" 데이터셋이며, AI Hub 이용약관을 따릅니다.
**저장소에는 영상 데이터를 포함하지 않습니다** (`.gitignore`로 제외).
재현하려면 AI Hub에서 직접 다운로드 후 `data/raw/01_real_word_video.zip` 으로 배치하면 됩니다.
