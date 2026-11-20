# Compression-tournament trainer (G.O.D SN56). Fixed track: compressed-tensors pack-quantized checkpoints.
# Freeform track: zstd-coded low-bit codes, expanded by decoder/decode.py (copied in for the round-trip self-check).
# Base: torch 2.8.0 + CUDA 12.8.1 (the text tournament's base; CUDA <= 12.8 on the validator's H100 hosts).
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN pip install --no-cache-dir \
        transformers==5.12.1 \
        accelerate==1.14.0 \
        safetensors==0.6.2 \
        tokenizers==0.22.1 \
        jinja2==3.1.6 \
        numpy==2.1.3 \
        zstandard==0.23.0 \
    && python -c "import torch, transformers; assert torch.__version__.startswith('2.8.0'), torch.__version__; print(torch.__version__, torch.version.cuda, transformers.__version__)"

RUN mkdir -p /app/checkpoints /workspace
WORKDIR /workspace
COPY quant /workspace/quant
COPY decoder /workspace/decoder
COPY LICENSE.md NOTICE /workspace/

# Self-check of the packer (CPU): dense bitstream round trip for 2/3/4 bits.
RUN python -c "import torch; from quant.packing import pack_last_dim, unpack_last_dim; \
[(lambda b,x: torch.equal(unpack_last_dim(pack_last_dim(x,b),b,x.shape[1]),x) or (_ for _ in ()).throw(AssertionError(b)))(b, torch.randint(0,1<<b,(7,384))) for b in (2,3,4)]; print('packing ok')"

ENTRYPOINT ["python", "-m", "quant.main"]
