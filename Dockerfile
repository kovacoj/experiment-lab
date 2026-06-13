FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md AGENTS.md .gitignore ./
COPY app ./app
COPY tests ./tests

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

CMD ["python", "-m", "app.labs.runner", "--scenario", "reputation_monitor"]
