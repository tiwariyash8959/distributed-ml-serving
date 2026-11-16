"""gRPC worker implementation."""

from __future__ import annotations

import logging
from concurrent import futures
from time import perf_counter

import grpc

from ml_serving.model import MODEL_VERSION, InvalidFeatures, predict
from ml_serving.observability import configure_logging
from ml_serving.v1 import prediction_pb2, prediction_pb2_grpc

logger = logging.getLogger("ml_serving.worker")


class PredictionWorker(prediction_pb2_grpc.PredictionServiceServicer):
    """Serve predictions from the local baseline model."""

    def Predict(self, request, context):
        if not request.request_id:
            logger.warning("prediction_rejected", extra={"event_fields": {"reason": "missing_request_id"}})
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request_id is required")
        if request.model_version and request.model_version != MODEL_VERSION:
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
            label, scores = predict(request.features)
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
                    "model_version": MODEL_VERSION,
                    "predicted_label": label,
                    "latency_ms": round((perf_counter() - started) * 1000, 3),
                }
            },
        )
        return prediction_pb2.PredictResponse(
            request_id=request.request_id,
            model_version=MODEL_VERSION,
            scores=scores,
            predicted_label=label,
        )

    def GetHealth(self, request, context):
        return prediction_pb2.HealthResponse(ready=True, model_version=MODEL_VERSION)


def create_server(bind_address: str = "127.0.0.1:50051", max_workers: int = 8) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    prediction_pb2_grpc.add_PredictionServiceServicer_to_server(PredictionWorker(), server)
    if server.add_insecure_port(bind_address) == 0:
        raise ValueError(f"could not bind gRPC worker to {bind_address}")
    return server


def serve(bind_address: str = "127.0.0.1:50051") -> None:
    configure_logging()
    server = create_server(bind_address)
    server.start()
    logger.info(
        "worker_started",
        extra={"event_fields": {"address": bind_address, "model_version": MODEL_VERSION}},
    )
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=3).wait()
