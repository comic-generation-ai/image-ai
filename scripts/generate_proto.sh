#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "$PROJECT_ROOT" || exit

PROTO_DIR="$PROJECT_ROOT/proto"
OUTPUT_DIR="$PROJECT_ROOT/src/service/generated"

echo "=== ĐANG BIÊN DỊCH GRPC PROTOBUF CHO IMAGE-AI ==="
echo "Thư mục Proto: $PROTO_DIR"
echo "Thư mục Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# 1. *.py (Mã hóa tin nhắn)
# 2. *.pyi (Type hint stubs hỗ trợ nhắc lệnh / IDE Autocomplete)
# 3. *_grpc.py (Các Stub và Servicer cho gRPC)
# Ưu tiên venv của project (grpcio-tools cài trong env/, không phải python3 hệ thống)
if [[ -x "$PROJECT_ROOT/env/bin/python3" ]]; then
    PYTHON="$PROJECT_ROOT/env/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo " Lỗi: Không tìm thấy python3. Chạy: python3 -m venv env && source env/bin/activate && pip install -r requirements.txt"
    exit 1
fi

echo "Đang biên dịch Protobuf (python: $PYTHON)..."
"$PYTHON" -m grpc_tools.protoc \
    -I"$PROTO_DIR" \
    --python_out="$OUTPUT_DIR" \
    --pyi_out="$OUTPUT_DIR" \
    --grpc_python_out="$OUTPUT_DIR" \
    "$PROTO_DIR/image_generation.proto"

if [ $? -eq 0 ]; then
    echo " Biên dịch Protobuf thành công!"
    echo "Đã tạo 3 tệp tin tại $OUTPUT_DIR:"
    echo "  - image_generation_pb2.py"
    echo "  - image_generation_pb2.pyi"
    echo "  - image_generation_pb2_grpc.py"

    # macOS và Linux có sự khác biệt nhỏ về cú pháp sed -i
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # Hệ điều hành macOS
        sed -i '' 's/import image_generation_pb2/from . import image_generation_pb2/g' "$OUTPUT_DIR/image_generation_pb2_grpc.py"
    else
        # Hệ điều hành Linux / Ubuntu
        sed -i 's/import image_generation_pb2/from . import image_generation_pb2/g' "$OUTPUT_DIR/image_generation_pb2_grpc.py"
    fi

    echo " Đã sửa đường dẫn import tương đối (Python 3 style) thành công."
else
    echo " Lỗi: Biên dịch gRPC Protobuf thất bại!"
    exit 1
fi