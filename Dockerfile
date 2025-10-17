# 1. 베이스 이미지 선택 (Python 3.9)
FROM python:3.9-slim

# 2. 시스템 라이브러리 설치 (OpenCV 및 git에 필요)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 파이썬 라이브러리 설치 (가장 먼저 수행하여 Docker 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 모델 가중치 미리 다운로드
# 이미지를 빌드할 때 CLIP 모델을 미리 다운로드하여 이미지에 포함
# 이렇게 하면 나중에 컨테이너를 실행할 때마다 다운로드할 필요가 없어짐
RUN python -c "import clip; clip.load('ViT-B/32')"

# 6. 소스 전체 복사 (model, main.py, requirements.txt, test_video.mp4 포함)
COPY . .

# 7. 컨테이너가 시작될 때 실행할 기본 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]