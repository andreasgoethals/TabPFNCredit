"""Host-side container launcher. Only standard-library dependencies are required."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments) or ["--help"]
    repo = Path(__file__).resolve().parents[1]
    data = Path(os.environ["VSC_DATA"]).resolve(strict=True)
    if not repo.is_relative_to(data):
        raise SystemExit("The VSC checkout must reside on DATA")
    if "TABPFN_STAGING_ROOT" in os.environ:
        raw = os.environ["TABPFN_STAGING_ROOT"].strip()
    elif "TABPFN_PROJECT_ROOT" in os.environ or "VSC_PROJECT_ROOT" in os.environ:
        shared = os.environ.get("TABPFN_PROJECT_ROOT", os.environ.get("VSC_PROJECT_ROOT", "")).strip()
        if not shared:
            raise SystemExit("The shared PROJECT root cannot be empty")
        raw = str(Path(shared) / "TabPFNCredit")
    else:
        raw = "/lustre1/project/stg_00211/TabPFNCredit"
    if not raw or not Path(raw).is_absolute():
        raise SystemExit("An absolute PROJECT directory is required")
    project = Path(raw).resolve(strict=True)
    image = Path(os.environ.get("TABPFN_CONTAINER_IMAGE", str(project / "runtimes" /
                  "tabpfncredit-runtime-20261029-cli-fixed.sif"))).resolve(strict=True)
    if not image.is_file() or not image.is_relative_to(project):
        raise SystemExit("The versioned SIF must reside on PROJECT")
    scratch = Path(os.environ["VSC_SCRATCH"]).resolve(strict=True)
    native = Path(os.environ["VSC_SCRATCH_LUSTRE1"]).resolve(strict=True)
    if scratch != native:
        raise SystemExit("This deployment uses wICE and native Lustre scratch")
    parent = scratch / "tabpfncredit" / "launchers"
    if not parent.resolve().is_relative_to(scratch):
        raise SystemExit("Launcher workspace escapes scratch")
    parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="launch-", dir=parent))
    records = repo / "logs" / "manifests" / "submissions"
    records.mkdir(parents=True, exist_ok=True)
    manifest = records / f"{uuid.uuid4().hex}.json"
    (workspace / "OWNER.json").write_text(json.dumps({"purpose": "container launcher",
        "pid": os.getpid(), "manifest": str(manifest), "checkout": str(repo)}))
    scheduling = arguments[0] in ("experiment", "resubmit")
    preview = "--no-submit" in arguments or "--help" in arguments or "-h" in arguments
    if scheduling:
        arguments = [arg for arg in arguments if arg not in ("--submit", "--no-submit")]
        arguments.append("--no-submit")
    env = {key: value for key, value in os.environ.items() if not key.startswith("APPTAINERENV_")}
    settings = {key: value for key, value in env.items() if key.startswith("TABPFN_")}
    settings.update({
        "TABPFN_VSC_PROFILE": "container", "TABPFN_STAGING_ROOT": str(project),
        "TABPFN_CONTAINER_IMAGE": str(image),
        "TABPFN_CONTAINER_PYTHON": os.environ.get("TABPFN_CONTAINER_PYTHON", "python"),
        "TABPFN_RESULTS_ROOT": os.environ.get("TABPFN_RESULTS_ROOT", str(project / "results")),
        "TABPFN_CACHE_ROOT": str(workspace / "cache"), "TMPDIR": str(workspace / "tmp"),
        "VSC_DATA": str(data), "VSC_SCRATCH": str(scratch),
        "VSC_SCRATCH_LUSTRE1": str(native),
        "VSC_PROJECT_LUSTRE1": os.environ["VSC_PROJECT_LUSTRE1"],
        "VSC_HOME": os.environ.get("VSC_HOME", "VSC"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TABPFN_HOST_JOB_MANIFEST": str(manifest),
        "HF_HOME": str(project / "checkpoints" / "huggingface"),
        "TORCH_HOME": str(project / "checkpoints" / "torch"),
        "XDG_CACHE_HOME": str(workspace / "xdg"),
        "TRITON_CACHE_DIR": str(workspace / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(workspace / "inductor"),
        "TORCH_EXTENSIONS_DIR": str(workspace / "extensions"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TABPFN_MODEL_CACHE_DIR": str(project / "checkpoints" / "tabpfn"),
    })
    for name, value in settings.items():
        env[name] = value
        env["APPTAINERENV_" + name] = value
    for path in (workspace / "cache", workspace / "tmp", workspace / "apptainer"):
        path.mkdir()
    env["APPTAINER_CACHEDIR"] = str(workspace / "apptainer")
    env["APPTAINER_TMPDIR"] = str(workspace / "tmp")
    command = ["apptainer", "exec", "--cleanenv", "--pwd", str(repo)]
    for root in (data, project, scratch):
        command.extend(["--bind", f"{root}:{root}"])
    command += [str(image), settings["TABPFN_CONTAINER_PYTHON"], "-m", "src.utils.cli", *arguments]
    completed = subprocess.run(command, env=env)
    if completed.returncode:
        print(f"Retained failed launcher workspace: {workspace}", file=sys.stderr)
        return completed.returncode
    if manifest.exists():
        jobs = json.loads(manifest.read_text())
        if not preview:
            accepted = []
            receipt = manifest.with_suffix(".receipt.json")

            def record_receipt(value):
                temporary = receipt.with_suffix(".tmp")
                temporary.write_text(json.dumps(value, indent=2))
                temporary.replace(receipt)

            for job in jobs:
                script = Path(job["path"]).resolve(strict=True)
                if not script.is_relative_to(repo / "scripts") or script.suffix != ".slurm":
                    raise SystemExit("Invalid generated script in submission manifest")
                args = ["sbatch", "--parsable"]
                if job["dependency"]:
                    args.append("--dependency=" + job["dependency"])
                args.append(str(script))
                record_receipt({"accepted": accepted, "inflight": job, "status": "submitting"})
                result = subprocess.run(args, capture_output=True, text=True)
                raw = result.stdout.strip()
                identifier = raw.split(";", 1)[0]
                if result.returncode or not identifier.isdigit():
                    record_receipt({"accepted": accepted,
                        "uncertain": job, "stdout": result.stdout, "stderr": result.stderr})
                    raise SystemExit("Submission stopped; reconcile the receipt before retrying")
                accepted.append({**job, "id": identifier})
                record_receipt({"accepted": accepted, "status": "accepted"})
                print(f"submitted {script.name} -> {identifier}")
            primary = [job["id"] for job in accepted if job["cluster"] == "wice"]
            chain = primary or [job["id"] for job in accepted]
            print("Final job id (chain):", ":".join(chain) or "(none)")
    else:
        # Empty manifests add no recovery information for help/list/no-work scans.
        manifest.unlink(missing_ok=True)
    shutil.rmtree(workspace)
    try:
        parent.rmdir()
        parent.parent.rmdir()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
