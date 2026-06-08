# TALENT-side fixes (apply in your TALENT fork)

These are bugs that live in **TALENT's own source**, not in TabPFNCredit. They
were diagnosed from the Experiment-0 SLURM logs (`results/experiment0/logs/`)
and confirmed against the TALENT source dump. TabPFNCredit installs TALENT from
your fork (`pip install --no-deps "TALENT @ git+.../TALENT@main"`), so apply each
patch there and re-install.

Each entry gives the file, the root cause, and the exact change. Line numbers
are approximate — search for the quoted code.

Priority order (by how many runs each unblocks):

| # | Bug | Methods unblocked | Effort |
|---|-----|-------------------|--------|
| 1 | MSE `[N,1]` vs `[N]` assertion | danets, switchtab, tabtransformer, ptarl, bishop, amformer — **all LGD** | 1 line |
| 2 | torch 2.6 `weights_only` | tabcaps (all), grownet (most) | ~6 lines |
| 3 | protogate Long target into soft-CE | protogate — **all PD** | 1 line |
| 4 | tabicl_v2 unknown kwarg | tabicl_v2 — **all PD** (Exp 2 & 3 need this!) | ~3 lines |
| 5 | catboost `task_type='GPU'` on CPU node | catboost — **all** | config |
| 6 | amformer `topk` k>tokens | amformer on narrow datasets (0003.axa) | 2 lines |
| 7 | mitra weights not staged | mitra — **all** | see §7 + pre-stage |
| — | TabPFN license / HyperFast / Mitra downloads | tabpfn_v2_5, tabpfn_v3, hyperfast, mitra | **TabPFNCredit side** — see `docs/VSC_RUN.md` |

---

## 1. Regression MSE shape assertion `torch.Size([N,1]) != torch.Size([N])`

**File:** `TALENT/model/lib/data.py` → `mse_safe_broadcast` (called from
`TALENT/model/methods/base.py::train_epoch`, `loss = self.criterion(self.model(...), y)`).

**Cause:** regression heads emit `[N, 1]`; the target is `[N]`. The guard
asserts equality instead of squeezing, so every deep regression method dies in
training on **all 7 LGD datasets**.

**Fix** — squeeze the singleton dim in the one choke point:

```python
def mse_safe_broadcast(input, target):
    # NEW: align [N,1] model output with [N] target before MSE.
    if input.dim() == 2 and input.shape[1] == 1 and target.dim() == 1:
        input = input.squeeze(1)
    assert input.shape == target.shape, f"{input.shape} != {target.shape}"
    return F.mse_loss(input, target)
```

(One edit fixes danets, switchtab, tabtransformer, ptarl, bishop, amformer at
once. The eval path already tolerates the mismatch — sklearn metrics squeeze —
so no other change is needed.)

---

## 2. torch ≥ 2.6 flipped `torch.load(weights_only=True)` by default

**Error:** `_pickle.UnpicklingError: Weights only load failed ... Unsupported
global: GLOBAL ...CapsuleClassifier` (tabcaps) and `... torch.nn.modules.sparse.Embedding`
(grownet). These checkpoints pickle whole modules/objects, which the new default
refuses.

**Fix:** add `weights_only=False` to every `torch.load` of a *trusted local*
checkpoint. Full blast radius found in the dump (all need the flag except the
limix one, which already has it):

| File | Code |
|------|------|
| `TALENT/model/lib/tabcaps/model/tabcaps_model.py` (`load_model`) | `torch.load(filepath)` → `torch.load(filepath, weights_only=False)` |
| `TALENT/model/models/grownet.py` (`from_file`) | `torch.load(path)` → `torch.load(path, weights_only=False)` |
| `TALENT/model/lib/tabcaps/lib/logger.py` | `torch.load(lastest_out_path)` → `..., weights_only=False)` |
| `TALENT/model/classical_methods/rfm.py` | `torch.load(ops.join(...))` → `..., weights_only=False)` |
| `TALENT/model/methods/ptarl.py` | `torch.load(osp.join(...))['params']` → `torch.load(..., weights_only=False)['params']` |
| `TALENT/model/methods/tabm.py` | same shape as ptarl |

---

## 3. protogate passes a Long target into a soft-label cross-entropy

**File:** `TALENT/model/methods/protogate.py` ~line 122 (`predict`):
`vl = self.criterion(test_logit, test_label).item()`.

**Error:** `RuntimeError: Expected floating point type for target with class
probabilities, got Long` — fails on **all PD datasets**.

**Cause:** `self.criterion` is a soft-label CE (expects float class-probability
targets) but `test_label` is integer class indices. (This surfaced after the
earlier "fix protogate vl=1.0" change started computing a real val loss.)

**Fix** — use index cross-entropy for the hard integer labels:

```python
# old:
vl = self.criterion(test_logit, test_label).item()
# new:
vl = F.cross_entropy(test_logit, test_label.long()).item()
```

(If protogate genuinely needs the soft-label loss elsewhere, instead one-hot the
target: `F.one_hot(test_label.long(), num_classes=test_logit.shape[-1]).float()`.)

---

## 4. tabicl_v2 passes `use_hierarchical` (and friends) that the installed `tabicl` rejects

**File:** `TALENT/model/methods/tabicl_v2.py` → `construct_model`,
`self.model = TabICLClassifier(**common, **classifier_extras)`.

**Error:** `TypeError: TabICLClassifier.__init__() got an unexpected keyword
argument 'use_hierarchical'` — fails on **all PD datasets**. This blocks
`tabicl_v2`, which is one of the four methods in **Experiments 2 and 3**, so it
matters a lot for the sweeps.

**Cause:** the wrapper hardcodes classifier knobs (`use_hierarchical`,
`softmax_temperature`, `average_logits`, `class_shift`) that don't all exist in
the installed `tabicl` version's `TabICLClassifier.__init__`.

**Fix (robust to version drift)** — filter to the kwargs the installed signature
actually accepts:

```python
import inspect
classifier_extras = dict(
    softmax_temperature=general.get('softmax_temperature', 0.9),
    average_logits=general.get('average_logits', True),
    use_hierarchical=general.get('use_hierarchical', True),
    class_shift=general.get('class_shift', True),
)
accepted = set(inspect.signature(TabICLClassifier.__init__).parameters)
common = {k: v for k, v in common.items() if k in accepted}
classifier_extras = {k: v for k, v in classifier_extras.items() if k in accepted}
self.model = TabICLClassifier(**common, **classifier_extras)
```

Alternatively pin `tabicl` to the version whose `TabICLClassifier` accepts these
knobs. The introspection guard is preferred (survives either direction).

---

## 5. catboost runs `task_type='GPU'` on the CPU partition

**Error:** `_catboost.CatBoostError: CUDA error 35: CUDA driver version is
insufficient for CUDA runtime version` — fails on **all datasets**.

**Cause:** catboost is a classical method scheduled on the **CPU** partition
(`batch_sapphirerapids`, no usable CUDA driver), but it is configured with
`task_type='GPU'`. catboost's own default is CPU, so something in TALENT's
catboost config sets GPU.

**Fix:** in TALENT's catboost config/default (search for `task_type` under
`TALENT/model/classical_methods/catboost.py` or its config JSON), set
`task_type='CPU'`, or make it conditional:

```python
import torch
params['task_type'] = 'GPU' if torch.cuda.is_available() else 'CPU'
```

(TabPFNCredit routes catboost to the CPU partition on purpose — it has no GPU
allocated — so CPU is correct here.)

---

## 6. amformer `torch.topk(k=num_per_group)` with k > available tokens

**File:** `TALENT/model/lib/amformer/blocks.py` (`MemoryBlock.forward`), the
`torch.topk(attn, dim=-1, k=self.num_per_group)` call.

**Error:** `RuntimeError: selected index k out of range` — on narrow datasets
(seen on `0003.axa`).

**Fix** — clamp k to the available dim and reuse it consistently:

```python
k = min(self.num_per_group, attn.shape[-1])
value, idx_original = torch.topk(attn, dim=-1, k=k)
idx = idx_original.unsqueeze(-1).repeat((1, 1, 1, 1, d // h))
vv = v.unsqueeze(-2).repeat((1, 1, 1, k, 1))      # use k, not self.num_per_group
xx_ = torch.gather(vv, 2, idx)
```

(If the downstream `gather_layer` Conv1d width also trips on a very narrow
dataset, the cleaner fix is to clamp `num_per_group <= token_num` at
construction in `MemoryBlock.__init__`.)

---

## 7. mitra config/weights not present

**File:** `TALENT/model/methods/mitra.py` already resolves the path via
`resolve_bundled_path(...)` with a relative fallback — good — but the actual
**weights + `config.json` are not bundled**, so `Mitra.from_pretrained` still
hits `FileNotFoundError: .../models_mitra/cls/config.json`.

**Fix options:**
1. Pre-download the Mitra weights into the bundled dir
   (`TALENT/model/models/models_mitra/{cls,reg}/`) — see the pre-stage step in
   `docs/VSC_RUN.md`.
2. Or fall back to the HF Hub id when the local dir is missing:

```python
model_path = resolve_bundled_path(f"model/models/models_mitra/{subdir}/")
if model_path is None or not os.path.exists(os.path.join(str(model_path), "config.json")):
    model_path = "autogluon/mitra-classifier" if not self.is_regression else "autogluon/mitra-regressor"
self.model = Mitra.from_pretrained(path=model_path, device="cpu").to(self.args.device)
```

(Confirm the exact hub id and that `from_pretrained` accepts a hub id — that
logic is in `TALENT/model/lib/mitra/tab2d.py`, which wasn't in the dump.)

---

### Not TALENT bugs — handled on the TabPFNCredit side

* **TabPFN v2.5 / v3 license + weight download**, **HyperFast figshare download**,
  **Mitra HF download** all fail because wICE compute nodes have **no outbound
  internet**. Fixed by pre-staging weights on the login node and running offline
  on the compute nodes — see `docs/VSC_RUN.md` and `scripts/prestage_models.sh`.
  (The SLURM prologue now exports `HF_HOME`/`HF_HUB_OFFLINE=1`/`TABPFN_MODEL_CACHE_DIR`
  pointing at a shared `$VSC_DATA` cache.)
