# Dockerfile — packages the model + server + exact environment into one sealed box.
# The same image runs identically on a laptop and on Hugging Face Spaces, because
# the model artifact ships inside the image rather than being fetched at runtime.

FROM python:3.11-slim

# 7860 is the port Hugging Face Spaces expects a Docker Space to listen on.
# It is declared as app_port in the README front matter; the two must agree.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    APP_PORT=7860

# Spaces runs containers as a non-root user; create one with UID 1000 and give
# it a real HOME so TensorFlow/Keras have somewhere writable for their caches.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Dependencies first: this layer is cached, so code edits do not re-download
# the (large) TensorFlow wheel on every rebuild.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code, the shared transform, the test fixture, and the model.
COPY preprocessing.py serve.py smoke_test.py sample_digit.json ./
COPY static/ ./static/
COPY models/ ./models/

RUN chown -R appuser:appuser /app

USER appuser
ENV HOME=/home/appuser \
    KERAS_HOME=/home/appuser/.keras

EXPOSE 7860

# start-period is generous: loading TensorFlow plus the model takes a while on
# a cold CPU-only container, and we do not want the host to kill it early.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4).status == 200 else 1)"

# 0.0.0.0 means "accept connections from outside the container", which the host
# needs in order to reach the service at all.
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "7860"]
