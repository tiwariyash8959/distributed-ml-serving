# Architecture: initial direction

## Design boundaries

- **Coordinator:** accepts prediction requests, selects a healthy worker, applies deadlines, and returns typed results.
- **Worker:** loads one model version, validates inputs, performs inference, and reports readiness.
- **Model registry:** stores immutable artifact versions and metadata; artifacts live in object storage rather than the database.
- **Metadata store:** tracks model versions and operational records. Cassandra is a candidate; schema and consistency needs will be documented before adoption.
- **Gateway:** optional HTTP interface for clients that do not use gRPC directly.

## Reliability decisions to resolve

1. Whether retries are safe for each RPC and how request IDs prevent duplicate side effects.
2. How readiness differs from liveness during model loading.
3. How model versions are selected and rolled back.
4. How the coordinator avoids routing overload to a worker that is technically healthy.
5. Which metrics and traces are required to explain tail latency and failed requests.

## First milestone: implemented locally

The worker loads a validated, versioned linear model from a JSON artifact. The default artifact ships with the Python package; operators can select another compatible artifact at startup. A Docker Compose setup runs two workers behind the coordinator and Prometheus scrapes RPC request counts and latency histograms from all three services. Artifacts and metadata are not yet stored in object storage or a database, and Kubernetes deployment remains future work.

## Coordinator behavior

The initial coordinator maintains one gRPC channel per configured worker and rotates the first target for each call. It retries only transport unavailability or worker-side deadline errors, within a shared request deadline. This is currently safe because inference has no side effects. Health reports ready when at least one worker responds ready.
