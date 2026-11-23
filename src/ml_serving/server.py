"""gRPC worker implementation."""

from __future__ import annotations

import logging
from concurrent import futures
from time import perf_counter

import grpc

from ml_serving.model import (
    DEFAULT_MODEL,
    InvalidFeatures,
    ModelArtifact,
    default_model_artifact_path,
    load_model_artifact,
)
from ml_serving.metrics import start_metrics_server, stop_metrics_server, track_rpc
from ml_serving.observability import configure_logging
from ml_serving.v1 import prediction_pb2, prediction_pb2_grpc

logger = logging.getLogger("ml_serving.worker")


class PredictionWorker(prediction_pb2_grpc.PredictionServiceServicer):
    """Serve predictions from a validated local model artifact."""

    def __init__(self, model: ModelArtifact = DEFAULT_MODEL) -> None:
        self._model = model

    @track_rpc("worker", "Predict")
    def Predict(self, request, context):
        if not request.request_id:
            logger.warning("prediction_rejected", extra={"event_fields": {"reason": "missing_request_id"}})
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request_id is required")
        if request.model_version and request.model_version != self._model.model_version:
            logger.warning(
                "prediction_rejected",
                extra={
                    "event_fields": {
                        "request_id": request.request_id,
                        "reason": "unknown_model_version",
                        "model_version": request.model_version,
                    }
                },
            )
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown model version: {request.model_version}")
        started = perf_counter()
        try:
            label, scores = self._model.predict(request.features)
        except InvalidFeatures as error:
            logger.warning(
                "prediction_rejected",
                extra={
                    "event_fields": {
                        "request_id": request.request_id,
                        "reason": "invalid_features",
                        "detail": str(error),
                    }
                },
            )
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))

        logger.info(
            "prediction_completed",
            extra={
                "event_fields": {
                    "request_id": request.request_id,
                    "model_version": self._model.model_version,
                    "predicted_label": label,
                    "latency_ms": round((perf_counter() - started) * 1000, 3),
                }
            },
        )
        return prediction_pb2.PredictResponse(
            request_id=request.request_id,
            model_version=self._model.model_version,
            scores=scores,
            predicted_label=label,
        )

    @track_rpc("worker", "GetHealth")
    def GetHealth(self, request, context):
        return prediction_pb2.HealthResponse(ready=True, model_version=self._model.model_version)


def create_server(
    bind_address: str = "127.0.0.1:50051",
    max_workers: int = 8,
    model_artifact_path: str | None = None,
) -> grpc.Server:
    model = (
        load_model_artifact(model_artifact_path)
        if model_artifact_path is not None
        else DEFAULT_MODEL
    )
    return _create_server(bind_address, max_workers, model)


def _create_server(bind_address: str, max_workers: int, model: ModelArtifact) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    prediction_pb2_grpc.add_PredictionServiceServicer_to_server(PredictionWorker(model), server)
    if server.add_insecure_port(bind_address) == 0:
        raise ValueError(f"could not bind gRPC worker to {bind_address}")
    return server


def serve(
    bind_address: str = "127.0.0.1:50051",
    model_artifact_path: str | None = None,
    metrics_address: str = "127.0.0.1",
    metrics_port: int | None = None,
) -> None:
    configure_logging()
    artifact_path = model_artifact_path or str(default_model_artifact_path())
    model = load_model_artifact(artifact_path)
    server = _create_server(bind_address, 8, model)
    metrics_state = start_metrics_server(metrics_address, metrics_port)
    try:
        server.start()
        logger.info(
            "worker_started",
            extra={
                "event_fields": {
                    "address": bind_address,
                    "model_version": model.model_version,
                    "model_artifact": artifact_path,
                    "metrics_address": (
                        f"{metrics_address}:{metrics_port}" if metrics_port is not None else None
                    ),
                }
            },
        )
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=3).wait()
    finally:
        stop_metrics_server(metrics_state)
