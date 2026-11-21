# Architecture-track image: both the trusted VM orchestrator and the sandbox
# that runs participant bundles. Participants get exactly these packages: the
# sandbox has no network, so nothing can be installed at scoring time.
# Build from the repository root and publish by digest:
#   docker build -f containers/arch.Dockerfile -t REGISTRY/delta-arch:TAG .
FROM vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
WORKDIR /app
# Keep the base image's CUDA torch build; FLA supplies the Gated DeltaNet kernels
# transformers uses for Qwen3.5 (otherwise it falls back to slow pure-torch code).
RUN python3 -c "import torch; print('torch==' + torch.__version__)" > /tmp/constraints.txt && \
    python3 -m pip install --no-cache-dir -c /tmp/constraints.txt transformers==5.22.1 tokenizers==0.23.3 \
      safetensors==0.8.0 flash-linear-attention==0.6.0 typer==0.16.0 httpx==0.28.1 filelock huggingface_hub numpy && \
    python3 -c "import fla, transformers, torch; print(torch.__version__, transformers.__version__, fla.__version__)"
# Installed (not PYTHONPATH): the runner starts with `python3 -I`, which ignores it.
COPY pyproject.toml README.md /src/
COPY delta_cli /src/delta_cli
RUN python3 -m pip install --no-cache-dir --no-deps /src && rm -rf /src
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
ENTRYPOINT []
CMD ["python3", "-m", "delta_cli.arch.worker"]
