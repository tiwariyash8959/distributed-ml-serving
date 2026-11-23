"""Prometheus metrics for gRPC request volume and latency."""

from __future__ import annotations

from functools import wraps
from time import perf_counter
from typing import Any

import grpc
from prometheus_client import Counter, Histogram, start_http_server

RPC_REQUESTS = Counter(
    "ml_serving_rpc_requests",
    "Completed gRPC requests by service, method, and gRPC status.",
    ("service", "method", "status"),
)
RPC_DURATION = Histogram(
    "ml_serving_rpc_duration_seconds",
    "Time spent handling gRPC requests.",
    ("service", "method"),
)


def track_rpc(service: str, method: str):
    """Record request count and duration for a gRPC servicer method."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any):
            started = perf_counter()
            status = "OK"
            context = kwargs.get("context") or (args[-1] if args else None)
            try:
                return function(*args, **kwargs)
            except grpc.RpcError as error:
                code = error.code()
                status = code.name if code is not None else "UNKNOWN"
                raise
            except Exception:
                status_code = context.code() if context is not None else None
                status = status_code.name if status_code is not None else "INTERNAL"
                raise
            finally:
                RPC_REQUESTS.labels(service, method, status).inc()
                RPC_DURATION.labels(service, method).observe(perf_counter() - started)

        return wrapped

    return decorate


def start_metrics_server(address: str, port: int | None):
    """Expose metrics on a separate HTTP endpoint when a port is configured."""
    if port is None:
        return None
    if not 1 <= port <= 65535:
        raise ValueError("metrics port must be between 1 and 65535")
    return start_http_server(port, addr=address)


def stop_metrics_server(server_state) -> None:
    """Stop a metrics endpoint started by ``start_metrics_server``."""
    if server_state is None:
        return
    http_server, thread = server_state
    http_server.shutdown()
    http_server.server_close()
    thread.join(timeout=1)
