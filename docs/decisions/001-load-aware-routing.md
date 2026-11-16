# Decision 001: choose workers by predicted completion time

## Context

Round robin distributes request counts evenly, but requests can overlap and workers may have different observed response times. Sending the next call to a worker that already has a long-running request can increase tail latency.

## Decision

The coordinator estimates each worker's completion time as:

```text
(coordinator-observed in-flight requests + 1) × EWMA worker-call latency
```

It reserves the worker with the lowest estimate. Ties rotate across configured endpoints. Latency uses an exponentially weighted moving average with alpha 0.2 and a 25 ms initial estimate. If an RPC fails with `UNAVAILABLE` or `DEADLINE_EXCEEDED`, the coordinator releases the reservation and tries another worker within the remaining request deadline.

## Why this choice

This needs no external metrics service and responds to both concurrent work and observed latency. It is small enough to reason about while the project has a single coordinator.

## Limits

The estimate sees only traffic routed through this coordinator; it does not include work sent by other coordinators or unrelated CPU load. A circuit breaker, shared worker telemetry, and controlled load comparisons are future work. The policy should be evaluated against round robin with repeatable concurrent benchmarks before claiming a performance improvement.
