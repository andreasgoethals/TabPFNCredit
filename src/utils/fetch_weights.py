#!/usr/bin/env python
# ============================================================================
# fetch_weights.py -- download ALL foundation-model weights into ./checkpoints
# ============================================================================
# Run this ONCE on a machine WITH internet (typically your local workstation).
# It downloads every tabular-foundation-model checkpoint the benchmark needs
# into a single, self-contained ``checkpoints/`` folder. You then upload that
# folder to the VSC (``$VSC_DATA/TabPFNCredit/checkpoints/``) where the compute
# nodes -- which have NO outbound internet -- read it OFFLINE.
#
#   python -m src.utils.fetch_weights            # download everything
#   python -m src.utils.fetch_weights --only tabpfn_v3 tabicl_v2   # a subset
#   python -m src.utils.fetch_weights --list     # show what would be fetched
#
# Why a dedicated script (instead of letting the models self-download on the
# cluster)?  wICE compute nodes have no internet, and the login node's network
# was flaky for large HF downloads. Fetching locally is reliable, resumable,
# and keeps the (multi-GB) weights out of the cluster's download path.
#
# Layout produced (mirrors exactly what the SLURM jobs expect via the env vars
# exported in src/utils/slurm_generator.py):
#
#   checkpoints/
#     huggingface/             # HF_HOME            -> TabICL, TabDPT (hub cache)
#       hub/...                # HUGGINGFACE_HUB_CACHE
#     tabpfn/                  # TABPFN_MODEL_CACHE_DIR -> TabPFN v2/v2.5/v3 .ckpt
#     torch/  xdg/             # TORCH_HOME / XDG_CACHE_HOME (parity; usually empty)
#     talent_assets/           # models that load from a PACKAGE-INTERNAL path,
#       models_mitra/cls/      #   NOT a cache -- setup_vsc_checkpoints.sh copies
#       models_mitra/reg/      #   these into the installed TALENT package / repo
#       hyperfast/             #   on the VSC after upload.
#
# After it finishes, follow the printed "NEXT STEPS".
# ============================================================================
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional

# ----------------------------------------------------------------------------
# Paths + cache env. We set the HF / Torch / XDG / TabPFN cache env vars to
# point INTO checkpoints/ *before* importing huggingface_hub, so the hub cache
# lands there. These are the SAME variable names the generated SLURM scripts
# export on the cluster, so "downloaded here" == "found there".
# ----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]   # src/utils/ -> repo root


def _configure_cache_env(checkpoints: Path) -> None:
    hf_home = checkpoints / "huggingface"
    env = {
        "HF_HOME": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "TORCH_HOME": str(checkpoints / "torch"),
        "XDG_CACHE_HOME": str(checkpoints / "xdg"),
        "TABPFN_MODEL_CACHE_DIR": str(checkpoints / "tabpfn"),
    }
    for key, value in env.items():
        os.environ[key] = value
    # We WANT to download here -- make sure no stray offline flag blocks us.
    for off in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(off, None)
    for sub in (
        hf_home / "hub",
        checkpoints / "tabpfn",
        checkpoints / "torch",
        checkpoints / "xdg",
        checkpoints / "talent_assets" / "models_mitra" / "cls",
        checkpoints / "talent_assets" / "models_mitra" / "reg",
        checkpoints / "talent_assets" / "hyperfast",
    ):
        sub.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Download spec table
# ----------------------------------------------------------------------------
# Each ``ModelSpec`` knows how to fetch one model's weights. ``method_names``
# lists the TALENT method names that consume it (so --only can match either a
# spec key or a method name). ``dest`` is one of:
#   "hub"     -> HF hub cache (HF_HOME). The model reads it back via
#                hf_hub_download(..., local_files_only=True) on the cluster.
#   "tabpfn"  -> checkpoints/tabpfn/<filename> (TABPFN_MODEL_CACHE_DIR). TabPFN
#                drops its .ckpt files here directly (not in hub layout).
#   "asset:<relpath>" -> checkpoints/talent_assets/<relpath>/ . These models
#                load from a package-internal / cwd-relative path, NOT a cache,
#                so setup_vsc_checkpoints.sh places them after upload.

class HFFile(NamedTuple):
    repo_id: str
    filename: str


class ModelSpec(NamedTuple):
    key: str
    method_names: tuple
    dest: str                  # "hub" | "tabpfn" | "asset:<relpath>"
    hf_files: tuple            # tuple[HFFile, ...]
    url: Optional[str] = None  # for non-HF direct downloads (HyperFast)
    url_filename: Optional[str] = None
    note: str = ""


# --- TabPFN default checkpoints --------------------------------------------
# Default filenames are read from the installed `tabpfn` package when possible
# (robust to upstream renames); this table is the fallback + documents intent.
_TABPFN_FALLBACK = {
    "tabpfn_v2": [
        HFFile("Prior-Labs/TabPFN-v2-clf", "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"),
        HFFile("Prior-Labs/TabPFN-v2-reg", "tabpfn-v2-regressor.ckpt"),
    ],
    "tabpfn_v2_5": [
        HFFile("Prior-Labs/tabpfn_2_5", "tabpfn-v2.5-classifier-v2.5_default.ckpt"),
        HFFile("Prior-Labs/tabpfn_2_5", "tabpfn-v2.5-regressor-v2.5_default.ckpt"),
    ],
    "tabpfn_v3": [
        HFFile("Prior-Labs/tabpfn_3", "tabpfn-v3-classifier-v3_default.ckpt"),
        HFFile("Prior-Labs/tabpfn_3", "tabpfn-v3-regressor-v3_default.ckpt"),
    ],
}


def _resolve_tabpfn_files() -> Dict[str, List[HFFile]]:
    """Read the *current* default ckpt filenames from the installed tabpfn
    package so we stay correct across upstream renames; fall back to the
    hard-coded table if the package or its internals are unavailable.
    """
    out: Dict[str, List[HFFile]] = {k: list(v) for k, v in _TABPFN_FALLBACK.items()}
    try:
        from tabpfn.constants import ModelVersion  # type: ignore
        from tabpfn.model_loading import ModelType, _get_model_source  # type: ignore

        mapping = {
            "tabpfn_v2": ModelVersion.V2,
            "tabpfn_v2_5": ModelVersion.V2_5,
            "tabpfn_v3": ModelVersion.V3,
        }
        for key, version in mapping.items():
            files: List[HFFile] = []
            for mtype in (ModelType.CLASSIFIER, ModelType.REGRESSOR):
                src = _get_model_source(version, mtype)
                files.append(HFFile(src.repo_id, src.default_filename))
            out[key] = files
    except Exception as exc:  # noqa: BLE001 -- best effort; fallback table is fine
        print(f"  [info] could not introspect tabpfn defaults ({exc!s}); "
              f"using built-in filename table.")
    return out


def _build_specs() -> List[ModelSpec]:
    tabpfn_files = _resolve_tabpfn_files()
    return [
        # ---- TabPFN family (-> checkpoints/tabpfn, via TABPFN_MODEL_CACHE_DIR) ----
        ModelSpec("tabpfn_v2", ("tabpfn_v2", "tabpfn_real"), "tabpfn",
                  tuple(tabpfn_files["tabpfn_v2"]),
                  note="TabPFN v2 (also serves tabpfn_real's finetuned clf ckpt)"),
        ModelSpec("tabpfn_v2_5", ("tabpfn_v2_5",), "tabpfn",
                  tuple(tabpfn_files["tabpfn_v2_5"]), note="TabPFN v2.5"),
        ModelSpec("tabpfn_v3", ("tabpfn_v3",), "tabpfn",
                  tuple(tabpfn_files["tabpfn_v3"]), note="TabPFN v3"),
        # NB: tabpfn (v1) loads a checkpoint BUNDLED inside the TALENT package
        # (prior_diff_real_checkpoint_n_0_epoch_42.cpkt) -- nothing to download.

        # ---- TabICL (-> HF hub cache, via HF_HOME) ----
        ModelSpec("tabicl", ("tabicl",), "hub",
                  (HFFile("jingang/TabICL-clf", "tabicl-classifier-v1.1-0506.ckpt"),),
                  note="TabICL v1.1 (TALENT's vendored lib; classifier-only)"),
        ModelSpec("tabicl_v2", ("tabicl_v2",), "hub",
                  (HFFile("jingang/TabICL", "tabicl-classifier-v2-20260212.ckpt"),
                   HFFile("jingang/TabICL", "tabicl-regressor-v2-20260212.ckpt")),
                  note="TabICL v2 (pip `tabicl` package; clf + reg)"),

        # ---- TabDPT (-> HF hub cache, via HF_HOME) ----
        ModelSpec("tabdpt", ("tabdpt",), "hub",
                  (HFFile("Layer6/TabDPT", "tabdpt1_1.safetensors"),),
                  note="TabDPT (Layer6)"),

        # ---- Mitra (-> talent_assets; loads from a package-internal dir) ----
        ModelSpec("mitra", ("mitra",), "asset:models_mitra/cls",
                  (HFFile("autogluon/mitra-classifier", "config.json"),
                   HFFile("autogluon/mitra-classifier", "model.safetensors")),
                  note="Mitra classifier (AutoGluon)"),
        ModelSpec("mitra_reg", ("mitra",), "asset:models_mitra/reg",
                  (HFFile("autogluon/mitra-regressor", "config.json"),
                   HFFile("autogluon/mitra-regressor", "model.safetensors")),
                  note="Mitra regressor (AutoGluon)"),

        # ---- HyperFast (-> talent_assets; downloads from figshare to a path) ----
        ModelSpec("hyperfast", ("hyperfast",), "asset:hyperfast",
                  tuple(), url="https://figshare.com/ndownloader/files/43484094",
                  url_filename="hyperfast.ckpt", note="HyperFast (figshare)"),
    ]


# ----------------------------------------------------------------------------
# Downloaders
# ----------------------------------------------------------------------------

def _hf_download(file: HFFile, *, dest: str, checkpoints: Path, token: Optional[str]) -> Path:
    from huggingface_hub import hf_hub_download

    if dest == "hub":
        # No local_dir -> lands in the HF hub cache (HF_HOME/hub). The models
        # read it back with local_files_only=True on the offline cluster.
        path = hf_hub_download(repo_id=file.repo_id, filename=file.filename, token=token)
    elif dest == "tabpfn":
        # TabPFN expects <TABPFN_MODEL_CACHE_DIR>/<filename> directly.
        path = hf_hub_download(repo_id=file.repo_id, filename=file.filename,
                               local_dir=str(checkpoints / "tabpfn"), token=token)
    elif dest.startswith("asset:"):
        rel = dest.split(":", 1)[1]
        out_dir = checkpoints / "talent_assets" / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(repo_id=file.repo_id, filename=file.filename,
                               local_dir=str(out_dir), token=token)
    else:
        raise ValueError(f"unknown dest {dest!r}")
    return Path(path)


def _url_download(url: str, out_path: Path, *, attempts: int = 6, backoff: float = 5.0) -> Path:
    """Stream a URL to ``out_path``. Retries on figshare's ``202 Accepted``
    (it stages the file asynchronously and serves it on a later request).
    Raises with a manual-download hint if it never returns a body.
    """
    import time

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [skip] already present: {out_path.name}")
        return out_path
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    last_status = None
    for attempt in range(attempts):
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 tabpfncredit-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (trusted URL)
            last_status = resp.status
            # 202 = figshare is still preparing the file; wait and retry.
            if resp.status == 202:
                print(f"  [wait] server staging file (HTTP 202); retry {attempt + 1}/{attempts}...")
                time.sleep(backoff)
                continue
            total = int(resp.headers.get("Content-Length") or 0)
            read = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    read += len(chunk)
                    if total:
                        print(f"\r  downloading {out_path.name}: {read >> 20} / "
                              f"{total >> 20} MiB ({100.0 * read / total:4.1f}%)",
                              end="", flush=True)
            print()
            if read > 0:
                tmp.replace(out_path)
                return out_path
        time.sleep(backoff)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(
        f"could not download after {attempts} attempts (last HTTP status {last_status}). "
        f"Download it manually from:\n        {url}\n"
        f"      and place it at:\n        {out_path}"
    )


def _human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if f < 1024 or unit == "GiB":
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} GiB"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download foundation-model weights into ./checkpoints for offline VSC use.",
    )
    parser.add_argument(
        "--checkpoints-dir", default=os.environ.get("TABPFN_CHECKPOINTS_DIR"),
        help="Where to write the checkpoints (default: <repo>/checkpoints).",
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Only fetch these models (spec keys or TALENT method names, "
             "e.g. tabpfn_v3 tabicl_v2 mitra).",
    )
    parser.add_argument(
        "--skip", nargs="*", default=None, help="Skip these models (same matching as --only).",
    )
    parser.add_argument("--list", action="store_true", help="List the download plan and exit.")
    parser.add_argument(
        "--hf-token", default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (only needed if a repo is gated; none currently are).",
    )
    args = parser.parse_args(argv)

    checkpoints = Path(args.checkpoints_dir).resolve() if args.checkpoints_dir else _REPO_ROOT / "checkpoints"

    def _selected(spec: ModelSpec) -> bool:
        names = {spec.key, *spec.method_names}
        if args.only and not (names & set(args.only)):
            return False
        if args.skip and (names & set(args.skip)):
            return False
        return True

    specs = [s for s in _build_specs() if _selected(s)]

    print("=" * 74)
    print(f"TabPFNCredit weight fetcher -> {checkpoints}")
    print("=" * 74)
    if args.list:
        for s in specs:
            tgt = ("hub cache" if s.dest == "hub"
                   else "checkpoints/tabpfn" if s.dest == "tabpfn"
                   else "checkpoints/talent_assets/" + s.dest.split(':', 1)[1])
            srcs = [f"{f.repo_id}:{f.filename}" for f in s.hf_files] or [s.url or ""]
            print(f"  {s.key:14s} -> {tgt:34s} {s.note}")
            for src in srcs:
                print(f"      {src}")
        return 0

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("ERROR: huggingface_hub is required. Install the project first "
              "(pip install -e \".[local]\"), then re-run.", file=sys.stderr)
        return 2

    # Point the HF / TabPFN caches into checkpoints/ before any download.
    _configure_cache_env(checkpoints)

    ok: List[str] = []
    failed: List[str] = []
    for spec in specs:
        print(f"\n--- {spec.key}: {spec.note} ---")
        try:
            if spec.url:  # direct (non-HF) download, e.g. HyperFast/figshare
                rel = spec.dest.split(":", 1)[1]
                out = checkpoints / "talent_assets" / rel / (spec.url_filename or "model.bin")
                _url_download(spec.url, out)
            for f in spec.hf_files:
                dest_path = _hf_download(f, dest=spec.dest, checkpoints=checkpoints, token=args.hf_token)
                print(f"  [ok] {f.repo_id}:{f.filename}  ({_human(dest_path.stat().st_size)})")
            ok.append(spec.key)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {spec.key}: {type(exc).__name__}: {exc}")
            failed.append(spec.key)

    # ---- Summary + next steps -------------------------------------------------
    print("\n" + "=" * 74)
    print(f"Done. ok={len(ok)}  failed={len(failed)}")
    if failed:
        print(f"  FAILED: {', '.join(failed)} -- re-run (downloads resume) or check the error above.")
    try:
        print(f"  checkpoints/ total size: {_human(_dir_size(checkpoints))}")
    except Exception:  # noqa: BLE001
        pass
    print("=" * 74)
    print(dedent_next_steps())
    return 1 if failed else 0


def dedent_next_steps() -> str:
    return (
        "NEXT STEPS\n"
        "----------\n"
        "1. Upload the whole 'checkpoints/' folder to the VSC, into the repo root:\n"
        "       rsync -av checkpoints/ <vsc>:$VSC_DATA/TabPFNCredit/checkpoints/\n"
        "   (or scp -r). It is large (several GB) and gitignored on purpose.\n"
        "2. On a VSC LOGIN node, provision the package-internal models ONCE:\n"
        "       cd $VSC_DATA/TabPFNCredit && bash scripts/setup_vsc_checkpoints.sh\n"
        "   (copies Mitra + HyperFast weights to where their loaders expect them).\n"
        "3. Submit jobs as usual -- the generated SLURM scripts point HF_HOME /\n"
        "   TABPFN_MODEL_CACHE_DIR at checkpoints/ and set HF_HUB_OFFLINE=1, so the\n"
        "   compute nodes load every weight offline.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
