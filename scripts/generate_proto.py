"""Generate Python gRPC bindings from the versioned service definition."""

from pathlib import Path

from grpc_tools import protoc

ROOT = Path(__file__).resolve().parents[1]
PROTO_ROOT = ROOT / "proto"
PROTO_FILE = PROTO_ROOT / "ml_serving" / "v1" / "prediction.proto"
OUTPUT_ROOT = ROOT / "src"
GENERATED_PACKAGE = OUTPUT_ROOT / "ml_serving" / "v1"


def main() -> None:
    GENERATED_PACKAGE.mkdir(parents=True, exist_ok=True)
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{PROTO_ROOT}",
            f"--python_out={OUTPUT_ROOT}",
            f"--grpc_python_out={OUTPUT_ROOT}",
            str(PROTO_FILE),
        ]
    )
    if result != 0:
        raise SystemExit(result)


if __name__ == "__main__":
    main()
