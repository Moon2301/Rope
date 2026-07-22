# Rope Technical Documentation

This directory is a map for developers and AI agents working on Rope. It documents the **code that exists on the current branch**. Planned work is kept separately in [PERFORMANCE_BACKLOG.md](PERFORMANCE_BACKLOG.md) so proposed behavior is never confused with runtime behavior.

## Reading guide

- [ARCHITECTURE.md](ARCHITECTURE.md): entrypoint, modules, ownership, threads, and the GPU pipeline.
- [FLOWS.md](FLOWS.md): startup, media selection, face detection and assignment, preview, scrubbing, recording, and Auto Job flows.
- [STATE_AND_DATA.md](STATE_AND_DATA.md): state machines, manifests, caches, parameters, and invalidation rules.
- [CODE_RULES.md](CODE_RULES.md): mandatory rules for modifying the current codebase.
- [PERFORMANCE_BACKLOG.md](PERFORMANCE_BACKLOG.md): verified hotspots, priorities, target designs, and acceptance criteria.
- [MODEL_CATALOG.md](MODEL_CATALOG.md): exact model files, feature gates, shapes, providers, and missing-file behavior.
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md): symptom-driven diagnosis and recovery steps.
- [DEVELOPMENT.md](DEVELOPMENT.md): setup, tests, smoke checks, profiling, and contribution workflow.
- [API_CONTRACTS.md](API_CONTRACTS.md): signal payloads, frame/tensor contracts, and internal APIs.

## Current scope

- The primary UI is Qt/PySide6; the entrypoint is [`Rope.py`](../Rope.py).
- One `VideoManager` owns the active media session, playback, scrubbing, swapping, and recording.
- Models are loaded lazily through `Models`; backends can use ONNX Runtime CUDA or TensorRT EP.
- An Auto Job processes one video, one target slot, and one source embedding. The user must review segments before rendering.
- Temporal stabilization uses five landmarks and applies only to video preview, manual recording, and Auto Job rendering when enabled.
- Segment scanning and rendering are independent phases: tracking is a render parameter and does not invalidate the scan cache.

## Documentation maintenance rules

When architecture or data formats change:

1. Update these documents in the same commit as the code.
2. Link to classes and functions instead of line numbers because line numbers change frequently.
3. Clearly label content as `Current behavior`, `Proposal`, or `Completed`.
4. When a manifest or cache schema changes, increment its version and document migration or fallback behavior.
5. Never store secrets, developer-specific paths, or model binaries in `.ai`.
