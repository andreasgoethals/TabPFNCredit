# Agent instructions

## Shared VSC Library

`vsc-library/` is a pinned, entirely read-only infrastructure submodule. Read its
README, source snapshot, guides and Slurm examples; never edit, refresh, lint,
format or test inside it. There is no PROJECT_SPECIFIC exception. Keep this
project's VSC choices and operating instructions in `docs/VSC.md`, outside the
submodule. Shared corrections belong in the standalone VSC Library checkout.
Record the library pin when operational evidence depends on it.
