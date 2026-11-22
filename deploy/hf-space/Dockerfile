# MMP-Quant on a Hugging Face Space (Docker SDK, free CPU). Assembled by
# tools/build_hf_space.py -- do not build from the repo root directly.
FROM python:3.11-slim

# ffmpeg: audio extraction + frame sampling; libgl1/libglib2.0-0: OpenCV (RapidOCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    MMPQUANT_CONFIG=config.yaml \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only torch first: the default Linux wheel pulls ~2.5 GB of CUDA libraries
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user . .
RUN pip install --no-cache-dir --user ".[webapp]" \
    "en_core_web_trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl"

# Download every model at build time, so the first upload does not wait for it
RUN python warmup.py

EXPOSE 7860
CMD ["mmpquant", "serve", "--host", "0.0.0.0", "--port", "7860"]
