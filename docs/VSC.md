# TabPFNCredit on VSC

Shared infrastructure reference: [VSC Library](../vsc-library/README.md). The parent Git commit pins its revision; inspect it with `git submodule status -- vsc-library`.
Its [official repository snapshot](<../vsc-library/VSC Repository.txt>),
[guides](../vsc-library/guides/) and [Slurm templates](<../vsc-library/slurm templates/README.md>)
are read-only. Keep this project's actual paths, environment, resource choices,
submission/resume steps and run evidence in this file or the project's own run logs.
Check [upstream review flags](../vsc-library/UPSTREAM_CHANGES.md) and the
[live Leuven status](https://status.vscentrum.be/tier2_leuven.html) before applying a changed policy.

Project commands and environment options remain in [VSC_RUN.md](../VSC_RUN.md).
That older runbook contains Genius-era variants. Current Leuven P100/V100
partitions belong to wICE; do not select retired Genius routes without auditing
the project's generated job specification. This library migration does not
certify those older runtime variants. The default current wICE/Mindwell choices
and the actual application environment must be checked before submission.

Record the chosen partition, account variable, interpreter, persistent model
assets, native working filesystem, result location and resume identity here
when preparing the next validated deployment. Existing results and checkpoints
were not moved or rebuilt by this documentation migration.
