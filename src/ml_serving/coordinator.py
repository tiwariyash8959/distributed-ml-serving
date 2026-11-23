"""Load-aware gRPC coordinator with bounded worker failover."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent import futures
from dataclasses import dataclass

import grpc

from ml_serving.metrics import start_metrics_server, stop_metrics_server, track_rpc
from ml_serving.observability import configure_logging
from ml_serving.v1 import prediction_pb2, prediction_pb2_grpc

logger = logging.getLogger("ml_serving.coordinator")
_EWMA_ALPHA = 0.2
_INITIAL_LATENCY_MS = 25.0


@dataclass(slots=True)
class WorkerEndpoint:
    address: str
    channel: grpc.Channel
    stub: prediction_pb2_grpc.PredictionServiceStub
    in_flight: int = 0
    ewma_latency_ms: float = _INITIAL_LATENCY_MS


class PredictionCoordinator(prediction_pb2_grpc.PredictionServiceServicer):
    """Route predictions using local in-flight and EWMA latency observations."""

    def __init__(self, worker_addresses: list[str], request_timeout: float = 1.0) -> None:
        if not worker_addresses:
            raise ValueError("at least one worker address is required")
        if len(set(worker_addresses)) != len(worker_addresses):
            raise ValueError("worker addresses must be unique")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero")

        self._timeout = request_timeout
        self._workers = []
        for address in worker_addresses:
            channel = grpc.insecure_channel(address)
            stub = prediction_pb2_grpc.PredictionServiceStub(channel)
            self._workers.append(WorkerEndpoint(address, channel, stub))
        self._tie_cursor = 0
        self._lock = threading.Lock()

    def _reserve_worker(self, excluded: set[str]) -> WorkerEndpoint | None:
        """Reserve the least-loaded candidate, rotating ties for fairness."""
        with self._lock:
            eligible = [
                (index, worker)
                for index, worker in enumerate(self._workers)
                if worker.address not in excluded
            ]
            if not eligible:
                return None

            count = len(self._workers)
            cursor = self._tie_cursor
            index, selected = min(
                eligible,
                key=lambda item: (
                    (item[1].in_flight + 1) * item[1].ewma_latency_ms,
                    (item[0] - cursor) % count,
                ),
            )
            self._tie_cursor = (index + 1) % count
            selected.in_flight += 1
            return selected

    def _record_completion(self, worker: WorkerEndpoint, latency_ms: float) -> None:
        with self._lock:
            worker.in_flight = max(0, worker.in_flight - 1)
            observed_latency_ms = max(0.1, latency_ms)
            worker.ewma_latency_ms = (
                _EWMA_ALPHA * observed_latency_ms
                + (1 - _EWMA_ALPHA) * worker.ewma_latency_ms
            )

    def _remaining(self, context, started: float) -> float:
        remaining = max(0.0, self._timeout - (time.monotonic() - started))
        client_remaining = context.time_remaining()
        if client_remaining is not None:
            remaining = min(remaining, client_remaining)
        return remaining

    @track_rpc("coordinator", "Predict")
    def Predict(self, request, context):
        request_id = request.request_id
        if not request_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request_id is required")

        started = time.monotonic()
        attempted: set[str] = set()
        last_error = None
        attempt_number = 0

        while len(attempted) < len(self._workers):
            total_remaining = self._remaining(context, started)
            if total_remaining <= 0:
                logger.warning(
                    "prediction_deadline_exceeded",
                    extra={"event_fields": {"request_id": request_id, "attempts": attempt_number}},
                )
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "prediction deadline exceeded")

            candidates_left = len(self._workers) - len(attempted)
            worker = self._reserve_worker(attempted)
            if worker is None:
                break
            attempted.add(worker.address)
            attempt_number += 1
            attempt_timeout = total_remaining / candidates_left
            attempt_started = time.monotonic()

            logger.info(
                "prediction_dispatched",
                extra={
                    "event_fields": {
                        "request_id": request_id,
                        "worker": worker.address,
                        "attempt": attempt_number,
                        "timeout_ms": round(attempt_timeout * 1000, 2),
                        "worker_in_flight": worker.in_flight,
                        "worker_ewma_ms": round(worker.ewma_latency_ms, 2),
                    }
                },
            )

            try:
                response = worker.stub.Predict(request, timeout=attempt_timeout)
            except grpc.RpcError as error:
                latency_ms = (time.monotonic() - attempt_started) * 1000
                self._record_completion(worker, latency_ms)
                logger.warning(
                    "worker_attempt_failed",
                    extra={
                        "event_fields": {
                            "request_id": request_id,
                            "worker": worker.address,
                            "attempt": attempt_number,
                            "status": error.code().name,
                            "latency_ms": round(latency_ms, 2),
                        }
                    },
                )
                if error.code() not in (
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                ):
                    context.abort(error.code(), error.details() or "worker rejected prediction")
                last_error = error
                continue

            latency_ms = (time.monotonic() - attempt_started) * 1000
            self._record_completion(worker, latency_ms)
            logger.info(
                "prediction_completed",
                extra={
                    "event_fields": {
                        "request_id": request_id,
                        "worker": worker.address,
                        "attempt": attempt_number,
                        "latency_ms": round(latency_ms, 2),
                        "total_latency_ms": round((time.monotonic() - started) * 1000, 2),
                        "model_version": response.model_version,
                    }
                },
            )
            return response

        if self._remaining(context, started) <= 0:
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "prediction deadline exceeded")
        detail = last_error.details() if last_error is not None else "no workers available"
        context.abort(grpc.StatusCode.UNAVAILABLE, f"all workers failed: {detail}")

    @track_rpc("coordinator", "GetHealth")
    def GetHealth(self, request, context):
        started = time.monotonic()
        probe_id = str(uuid.uuid4())
        attempted: set[str] = set()

        while len(attempted) < len(self._workers):
            total_remaining = self._remaining(context, started)
            if total_remaining <= 0:
                break
            candidates_left = len(self._workers) - len(attempted)
            worker = self._reserve_worker(attempted)
            if worker is None:
                break
            attempted.add(worker.address)
            attempt_started = time.monotonic()
            try:
                health = worker.stub.GetHealth(
                    request,
                    timeout=total_remaining / candidates_left,
                )
            except grpc.RpcError as error:
                latency_ms = (time.monotonic() - attempt_started) * 1000
                self._record_completion(worker, latency_ms)
                logger.warning(
                    "health_probe_failed",
                    extra={
                        "event_fields": {
                            "probe_id": probe_id,
                            "worker": worker.address,
                            "status": error.code().name,
                            "latency_ms": round(latency_ms, 2),
                        }
                    },
                )
                continue

            latency_ms = (time.monotonic() - attempt_started) * 1000
            self._record_completion(worker, latency_ms)
            if health.ready:
                logger.info(
                    "health_probe_ready",
                    extra={
                        "event_fields": {
                            "probe_id": probe_id,
                            "worker": worker.address,
                            "model_version": health.model_version,
                            "latency_ms": round(latency_ms, 2),
                        }
                    },
                )
                return prediction_pb2.HealthResponse(
                    ready=True,
                    model_version=health.model_version,
                )

        logger.warning("health_probe_unready", extra={"event_fields": {"probe_id": probe_id}})
        return prediction_pb2.HealthResponse(ready=False)

    def close(self) -> None:
        for worker in self._workers:
            worker.channel.close()


def create_server(
    worker_addresses: list[str],
    bind_address: str = "127.0.0.1:50052",
    request_timeout: float = 1.0,
    max_workers: int = 8,
) -> tuple[grpc.Server, PredictionCoordinator]:
    coordinator = PredictionCoordinator(worker_addresses, request_timeout)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    prediction_pb2_grpc.add_PredictionServiceServicer_to_server(coordinator, server)
    if server.add_insecure_port(bind_address) == 0:
        coordinator.close()
        raise ValueError(f"could not bind coordinator to {bind_address}")
    return server, coordinator


def serve(
    worker_addresses: list[str],
    bind_address: str = "127.0.0.1:50052",
    request_timeout: float = 1.0,
    metrics_address: str = "127.0.0.1",
    metrics_port: int | None = None,
) -> None:
    configure_logging()
    server, coordinator = create_server(worker_addresses, bind_address, request_timeout)
    metrics_state = start_metrics_server(metrics_address, metrics_port)
    try:
        server.start()
        logger.info(
            "coordinator_started",
            extra={
                "event_fields": {
                    "address": bind_address,
                    "workers": worker_addresses,
                    "timeout_ms": round(request_timeout * 1000, 2),
                    "routing_policy": "least_predicted_completion_time",
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
        coordinator.close()
        stop_metrics_server(metrics_state)
