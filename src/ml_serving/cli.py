"""Command-line interface for the worker and its sample client."""

from __future__ import annotations

import argparse

from ml_serving.client import get_health, predict
from ml_serving.coordinator import serve as serve_coordinator
from ml_serving.server import serve


def _parse_features(value: str) -> list[float]:
    try:
        return [float(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("features must be comma-separated numbers") from error


def main() -> None:
    parser = argparse.ArgumentParser(prog="ml-serving")
    commands = parser.add_subparsers(dest="command", required=True)

    serve_parser = commands.add_parser("serve", help="start the gRPC prediction worker")
    serve_parser.add_argument("--address", default="127.0.0.1:50051")

    coordinator_parser = commands.add_parser(
        "coordinator", help="start a load-aware coordinator in front of workers"
    )
    coordinator_parser.add_argument("--address", default="127.0.0.1:50052")
    coordinator_parser.add_argument(
        "--workers",
        type=lambda value: [part.strip() for part in value.split(",") if part.strip()],
        required=True,
        help="comma-separated worker addresses, for example 127.0.0.1:50051,127.0.0.1:50053",
    )
    coordinator_parser.add_argument("--request-timeout", type=float, default=1.0)

    predict_parser = commands.add_parser("predict", help="send one prediction to a worker")
    predict_parser.add_argument("--address", default="127.0.0.1:50051")
    predict_parser.add_argument("--features", type=_parse_features, required=True)
    predict_parser.add_argument("--request-id", help="optional correlation ID for this request")

    health_parser = commands.add_parser("health", help="check worker or coordinator readiness")
    health_parser.add_argument("--address", default="127.0.0.1:50051")

    arguments = parser.parse_args()
    if arguments.command == "serve":
        serve(arguments.address)
    elif arguments.command == "coordinator":
        serve_coordinator(arguments.workers, arguments.address, arguments.request_timeout)
    elif arguments.command == "predict":
        response = predict(arguments.address, arguments.features, arguments.request_id)
        print(f"request_id: {response.request_id}")
        print(f"model_version: {response.model_version}")
        print(f"predicted_label: {response.predicted_label}")
        print("scores: " + ", ".join(f"{score:.4f}" for score in response.scores))
    else:
        response = get_health(arguments.address)
        print(f"ready: {str(response.ready).lower()}")
        if response.model_version:
            print(f"model_version: {response.model_version}")
        if not response.ready:
            raise SystemExit(1)
