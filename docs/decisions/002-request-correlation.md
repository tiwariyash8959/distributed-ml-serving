# Decision 002: propagate request IDs and emit JSON events

## Context

A prediction crosses the client, coordinator, and worker. Logs without a shared identifier make it difficult to connect routing decisions, retries, and inference outcomes.

## Decision

Clients generate a UUID request ID unless the caller supplies one. The ID is part of the protobuf request and response, and both coordinator and worker emit one-line JSON events with the same ID. Events include service, worker, attempt, status, model version, and elapsed time where available. Feature values are deliberately excluded from logs.

## Why this choice

It provides useful request-level tracing without adding an external tracing backend or changing the RPC flow. JSON lines are easy to inspect locally and can be ingested by common log systems later.

## Limits

This is log correlation, not distributed tracing: it has no spans, trace sampling, or context propagation standard. OpenTelemetry can be added when deployment and operational requirements are defined.
