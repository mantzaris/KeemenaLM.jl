# API Reference

This page is generated from exported docstrings in the package. It covers the
stable package surface for configuration, bundle IO, checkpointing, generation,
chat helpers, behavior scoring, and Flux/Lux backend entry points.

The v9 chatbot training scripts live under `tools/` and are documented in the
README and current chatbot docs rather than as exported package API.

```@autodocs
Modules = [KeemenaLM.Core, KeemenaLM.FluxBackend, KeemenaLM.LuxBackend]
Private = false
Order = [:type, :function]
```
