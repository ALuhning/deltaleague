# Architecture-track image: both the trusted VM orchestrator and the sandbox
# that runs participant bundles. Participants get exactly these packages: the
# sandbox has no network, so nothing can be installed at scoring time.
# Build from the repository root and publish by digest:
#   docker build -f containers/arch.Dockerfile -t REGISTRY/delta-arch:TAG .
FROM vllm/vllm-openai@sha256:c6e88abc61f250e3b0d72489b077107e92fd004c1270577700ef85dd3f41755d
WORKDIR /app
RUN python3 -m pip install --no-cache-dir transformers==5.22.1 safetensors==0.6.2 tokenizers==0.22.1 \
      flash-linear-attention==0.4.1 typer==0.16.0 httpx==0.28.1 filelock huggingface_hub numpy
# Installed (not PYTHONPATH): the runner starts with `python3 -I`, which ignores it.
COPY pyproject.toml README.md /src/
COPY delta_cli /src/delta_cli
RUN python3 -m pip install --no-cache-dir --no-deps /src && rm -rf /src
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
ENTRYPOINT []
CMD ["python3", "-m", "delta_cli.arch.worker"]
