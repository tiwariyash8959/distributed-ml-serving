"""Small command-line gRPC client for the local worker."""

from __future__ import annotations

import uuid

import grpc

from ml_serving.v1 import prediction_pb2, prediction_pb2_grpc


def predict(
    address: str,
    features: list[float],
    request_id: str | None = None,
) -> prediction_pb2.PredictResponse:
    with grpc.insecure_channel(address) as channel:
        stub = prediction_pb2_grpc.PredictionServiceStub(channel)
        return stub.Predict(
            prediction_pb2.PredictRequest(
                request_id=request_id or str(uuid.uuid4()),
                features=features,
            ),
            timeout=3.0,
        )


def get_health(address: str) -> prediction_pb2.HealthResponse:
    with grpc.insecure_channel(address) as channel:
        stub = prediction_pb2_grpc.PredictionServiceStub(channel)
        return stub.GetHealth(prediction_pb2.HealthRequest(), timeout=3.0)
