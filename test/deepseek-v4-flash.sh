export CUDA_VISIBLE_DEVICES=7
export FLASHINFER_CUDA_ARCH_LIST=9.0
export TORCH_CUDA_ARCH_LIST="9.0+PTX"
export SGLANG_DSV4_MODE=2604
export SGLANG_DSV4_2604_SUBMODE=2604B

numactl --interleave=all python -m sglang.launch_server \
  --host 0.0.0.0 --port 30000 \
  --model /data1/models/DeepSeek-V4-Flash \
  --kt-weight-path /data1/models/DeepSeek-V4-Flash \
  --kt-method MXFP4 \
  --kt-num-gpu-experts 30 \
  --kt-cpuinfer 60 \
  --kt-threadpool-count 2 \
  --kt-gpu-prefill-token-threshold 4096 \
  --kt-enable-dynamic-expert-update \
  --tensor-parallel-size 1 \
  --context-length 16384 \
  --attention-backend flashinfer \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 2048 \
  --max-running-requests 2 \
  --watchdog-timeout 1200 \
  --disable-shared-experts-fusion \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-radix-cache \
  --skip-server-warmup
