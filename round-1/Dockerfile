# Sử dụng vLLM OpenAI base image được cung cấp bởi BTC
FROM vllm/vllm-openai:v0.22.1

# Có thể cài đặt thêm các thư viện bổ sung (như flash-attn nâng cấp, v.v. nếu cần)
# RUN pip install --no-cache-dir <thư viện>

# Thiết lập biến môi trường tối ưu hóa
ENV PYTHONUNBUFFERED=1
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Copy các script bổ sung nếu cần tùy biến sâu entrypoint
# COPY ./src /app/src
