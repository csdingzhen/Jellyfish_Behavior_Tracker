# docs/ — LLM-Accessible Project Knowledge

This folder contains structured documentation for the Cassiopea pipeline, organised for retrieval by LLMs and new developers.

## Files

| File | Contents |
| --- | --- |
| [00_overview.md](00_overview.md) | Project goal, species biology, recording setup, what has been ruled out |
| [01_pipeline_architecture.md](01_pipeline_architecture.md) | Every pipeline stage: what it does, inputs, outputs, key parameters, key files |
| [02_codebase_map.md](02_codebase_map.md) | File-by-file guide — which file to edit for each concern |
| [03_design_decisions.md](03_design_decisions.md) | Why things are built this way; approaches tried and rejected |
| [04_performance.md](04_performance.md) | Hardware specs, timing tables, active optimisations, future opportunities |
| [05_flower_comparison.md](05_flower_comparison.md) | Mentor's independent FLOWER.py script: how it works, how it differs, combination strategies |

## How to use

For a new developer joining the project, read files in order: 00 → 01 → 02 → 03.

For an LLM asked to modify a specific part of the pipeline, `02_codebase_map.md` tells you which files are involved. `03_design_decisions.md` tells you why the code is structured the way it is and what not to change.

For performance questions, see `04_performance.md`. For understanding the mentor's independent approach and how to combine them, see `05_flower_comparison.md`.
