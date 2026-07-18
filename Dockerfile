# Sử dụng base image PyTorch có sẵn CUDA
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# noninteractive: thiếu dòng này apt-get kéo tzdata (transitive dep của
# libgl1-mesa-glx) sẽ bật prompt hỏi múi giờ — build không có TTY nên treo
# vô thời hạn, không lỗi rõ ràng, chỉ đứng im ở bước "Geographic area:".
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Biên dịch protobuf — khớp đường dẫn thật mà code import (src/service/generated/),
# giống scripts/generate_proto.sh dùng cho dev local.
RUN mkdir -p ./src/service/generated \
    && python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./src/service/generated \
    --pyi_out=./src/service/generated \
    --grpc_python_out=./src/service/generated \
    ./proto/image_generation.proto \
    && sed -i 's/import image_generation_pb2/from . import image_generation_pb2/g' ./src/service/generated/image_generation_pb2_grpc.py \
    && touch ./src/service/generated/__init__.py

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grpc; channel = grpc.insecure_channel('localhost:50051'); grpc.channel_ready_future(channel).result(timeout=5)" || exit 1

# Run server
CMD ["python", "-m", "src.server"]