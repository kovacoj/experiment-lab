"""Frontend bundle exporter.

Takes the lab engine's LabRunReport for `reputation_monitor` and emits the
JSON contracts described in the Signal Foundry PoC spec under
`frontend_bundle/`. Read-only consumer of `app/labs/runner.py` — does not
mutate fixtures or lab code.
"""
