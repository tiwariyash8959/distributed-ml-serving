# Distributed ML Serving

An original project exploring reliable, horizontally scalable model serving with gRPC and Kubernetes.

## Status

Early implementation. The baseline gRPC worker and load-aware coordinator are implemented. Manual smoke checks confirmed that a 32-request concurrent burst reached both workers and that prediction and readiness still work when the first configured worker is unavailable. These checks validate behavior, not performance; performance claims will wait for reproducible benchmarks.

## Goals

- Serve versioned model predictions through a typed gRPC API.
- Keep request coordination separate from model execution workers.
- Make worker discovery, timeouts, retries, and graceful shutdown explicit.
- Store model artifacts in object storage and keep metadata in a database.
- Run locally first, then deploy and observe the system on Kubernetes.
- Measure correctness, tail latency, throughput, and recovery behavior.

## Planned architecture

```text
Client -> API gateway -> Coordinator -> gRPC worker pool
                                  |          |
                                  |          +-> load model artifact from object storage
                                  +-> model metadata and request records in Cassandra
```

The first milestone will use a small, reproducible classifier and a local development setup. Cassandra and object storage will be introduced behind clear interfaces so the model-serving path can be developed and tested independently.

## Repository layout

```text
proto/                 Versioned gRPC service definitions
src/ml_serving/        Coordinator, worker, model, and storage packages
docs/                  Architecture decisions and operational notes
deploy/                Local and Kubernetes deployment configuration
```

## First working slice: gRPC worker

Requires Python 3.11+. Create and activate a virtual environment, install the project and development tools, generate the protobuf bindings, then start the worker:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts/generate_proto.py
ml-serving serve
```

In another terminal, send a sample request:

```powershell
ml-serving predict --features 0.8,0.2
```

The baseline is a deterministic three-class linear model intended to exercise the RPC path. It is not a trained production model. The worker exposes `Predict` and `GetHealth`; invalid input is returned as a gRPC status error.

## Versioned model artifacts

The worker loads a JSON model artifact at startup. The packaged `baseline-0.1` artifact is used by default. To load another compatible artifact, pass its path when starting the worker:

```powershell
ml-serving serve --model-artifact .\models\candidate.json
```

An artifact declares its `model_version`, `feature_count`, class `labels`, and a `weights` row for each label. Each row contains one coefficient per feature followed by the bias. The worker validates the file before binding its gRPC port and reports the loaded version in predictions and health responses. For example:

```json
{
  "model_version": "example-1.0",
  "feature_count": 2,
  "labels": ["class_a", "class_b"],
  "weights": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
}
```

Artifacts are local files in this milestone; object storage and persistent model metadata remain future work.

## Coordinator

Start two worker processes on separate ports, then start the coordinator in front of them:

```powershell
ml-serving serve --address 127.0.0.1:50051
ml-serving serve --address 127.0.0.1:50053
ml-serving coordinator --address 127.0.0.1:50052 --workers 127.0.0.1:50051,127.0.0.1:50053
```

Point the sample client at the coordinator with `ml-serving predict --address 127.0.0.1:50052 --features 0.8,0.2`. The coordinator chooses the worker with the lowest estimated completion time from local in-flight requests and observed EWMA latency; ties rotate. It retries another worker on transport unavailability. Retries are safe for this stateless inference operation. It divides the remaining request budget across remaining candidates, so a failed endpoint cannot consume all the time reserved for failover. Pass `--request-id <id>` to set a specific correlation ID. Coordinator and worker logs are JSON lines and carry that ID; feature values are not logged.

Check readiness with `ml-serving health --address 127.0.0.1:50052`. The coordinator is ready when any configured worker responds ready; it shares the health-check deadline across workers so an unreachable endpoint does not hide a healthy one.

## Roadmap

1. [x] Define the versioned prediction contract and project boundaries.
2. [x] Implement a baseline model worker and a gRPC prediction path.
3. [x] Add load-aware coordinator routing, deadlines, worker failover, and request-correlated JSON logs.
4. [x] Load validated, versioned model artifacts from local JSON files.
5. Add object artifact storage and model metadata persistence.
6. Containerize services and deploy to a local Kubernetes cluster.
7. Add observability, load tests, and documented benchmark results.
