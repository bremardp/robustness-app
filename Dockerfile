FROM python:3.11-slim

WORKDIR /app

# Corporate MITM proxies break pip's default TLS verification inside containers;
# --trusted-host is the common Docker + corporate-proxy workaround.
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --upgrade \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org pip

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries=5 \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -i ${PIP_INDEX_URL} \
    -r requirements.txt

COPY app ./app

ENV PYTHONPATH=/app
ENV MAX_UPLOAD_MB=50

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
