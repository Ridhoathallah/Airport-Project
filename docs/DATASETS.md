# Dataset Engineering & Fine-Tuning Data Infrastructure

> **Scope.** This document covers the **ML data lifecycle**: the data
> contract, dataset validation, leakage-safe splitting, dataset versioning,
> the metadata catalog, training-experiment provenance and reproducible
> evaluation.
>
> **Out of scope.** The runtime application (`app/`), the FastAPI surface and
> the operational monitoring database are owned by a separate engineering
> workstream. Nothing in this document changes them, and nothing in `ml/` is
> imported by the runtime.
> The dataset metadata catalog is a **different database** from the runtime
> monitoring database: different file, different schema, different
> lifecycle, no shared connection.

---

## 1. Verified repository and model structure

Facts below were read from the repository, not assumed.

| Item | Value | How it was verified |
|---|---|---|
| YOLOv5 revision | `v7.0-544-g20d1d78a`, commit `20d1d78a08277e365d57bfa3a2cce752772d9e59` (2026-08-02) | `git -C yolov5 describe --tags` |
| YOLOv5 source | vendored clone of `ultralytics/yolov5`, clean working tree | `git -C yolov5 status` |
| Base checkpoint | `models/yolov5s.pt`, 14,808,437 B, sha256 `8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee` | `shasum -a 256` |
| Checkpoint content | **stock COCO-pretrained YOLOv5s**: 80 classes, class 0 = `person`, `epoch = -1`, no EMA, dated 2022-02-08 | the `.pt` was loaded and `model.names` inspected |
| Runtime class filter | COCO class 0 only | `PERSON_CLASS_ID = 0`, `app/detect_people.py` |
| Frame rate | `TARGET_FPS = 30`, `CAPTURE_FPS = 30` | `app/detect_people.py` |
| Training entry point | `yolov5/train.py` (vendored) — **not** the modern `ultralytics` CLI | `yolov5/train.py` argparse |
| Installed environment | Python 3.9.6, torch 2.8.0, torchvision 0.23.0, numpy 2.0.2, opencv-python 5.0.0.93, pillow 11.3.0, PyYAML 6.0.3, ultralytics 8.4.120, ultralytics-thop 2.1.6, pandas 2.3.3, seaborn 0.13.2, scipy 1.13.1 | `importlib.metadata` |
| Accelerators | **no CUDA**; MPS available and functional | `torch.cuda.is_available()` → `False`; MPS conv verified |
| `yolov5/Arial.ttf` | **absent** — the first `train.py`/`val.py` run downloads it (upstream behaviour) | `ls yolov5/Arial.ttf` |

### 1.1 Existing image and video assets

| Path | Contents | Usable as dataset? |
|---|---|---|
| `input/` | **empty** | no source footage exists |
| `output/*.png` | 7 dashboard/HUD screenshots | no — not photographs of people, no annotations |
| `output/smoke_test/bus.jpg` | upstream smoke-test image | no |
| `yolov5/data/images/` | `bus.jpg`, `zidane.jpg` (upstream fixtures) | no |

**There is no dataset in this repository.** No images, no labels, no
annotation tooling, no manifests, no experiment logs and no checkpoints
beyond the upstream release. Every image in `output/` is a rendered UI
screenshot, and none of them is a photograph of a queue.

### 1.2 What exists today for ML

* `scripts/evaluate_counts.py` — an offline **occupancy-counting**
  evaluator (MAE, ±1/±2 accuracy, bias) of the runtime pipeline against
  manually annotated ground-truth intervals. It measures *counting*, not
  detection quality, and it is not an mAP evaluator.
* `yolov5/val.py` — the only detection-quality evaluator: precision,
  recall, mAP@0.5, mAP@0.5:0.95, per-class values.
* **No** dataset versioning, no split tooling, no annotation validation,
  no experiment tracking, no fine-tuning launcher.

### 1.3 Baseline test state

Before this workstream: `make test` → **137 passed**; `pytest tests/` →
**169 passed**. After: **202** and **234** respectively (the same 65 new
tests are collected by both runners). See §10 — these are historical
markers; run the suites for current totals (see [DEVELOPMENT.md](DEVELOPMENT.md)).

---

## 2. The data contract

Implemented in `ml/contract.py`; version `1.0.0`.

### 2.1 Layers, kept strictly separate

| Layer | What it is | Where it lives |
|---|---|---|
| Raw source assets | the original video / camera feed | `input/`, operator-managed |
| Curated images | frames extracted and normalised | `datasets/<version>/images/<split>/` |
| Reviewed annotations | human-confirmed YOLO label files | `datasets/<version>/labels/<split>/` |
| Manifest | immutable, hash-verified description of a dataset version | `datasets/manifests/<version>.json` |
| Splits | the train/val/test assignment inside a manifest | manifest → `splits` |
| Experiments | a resolved training configuration + its outcome | catalog `experiment` table |
| Checkpoints | model weights produced by an experiment | catalog `model_artifact` table |
| Collected examples | machine-proposed frames awaiting review | `datasets/collected/<run_id>/` |

### 2.2 Identifiers

| Identifier | Form | Derived from |
|---|---|---|
| `image_id` | `img_<16 hex>` | SHA-256 of the image **bytes** (content-addressed) |
| `annotation_id` | `ann_<16 hex>` | SHA-256 of (image_id, class_id, box) |
| `dataset_version_id` | `dsv_<16 hex>` | SHA-256 of the canonical manifest body |
| `experiment_id` | `exp_<utcstamp>_<8 hex>` | creation time + config hash |
| `run_id` | `run_<utcstamp>` | collection run timestamp |
| `manifest_sha256` | 64 hex | SHA-256 of the canonical manifest body |

Because images are content-addressed, two files with the same
`image_id` **are** duplicates — there is no separate "duplicate" concept
to keep in sync.

### 2.3 Timestamps

ISO-8601 UTC with a trailing `Z` and millisecond precision
(`2026-10-20T01:57:00.307Z`), matching the existing project convention
(`app/queue_prototype.py: iso_from_unix`).
Manifests store the string; the catalog stores the same string. The
runtime monitoring database's epoch-seconds convention is a different
schema and is not shared.

### 2.4 Provenance and honesty rules

Each image record may carry: `source_group`, `source_ref`, `source_kind`,
`captured_at_utc`, `license`, `usage_restrictions`, `annotation_status`,
`class_schema_version`, `preprocessing`, `review_note`.

* Missing provenance stays **missing** (`null`). It is never invented,
  defaulted, or back-filled with a plausible value.
* `annotation_status` is one of `verified`, `negative_reviewed`,
  `machine_proposed`, `unreviewed`. Only `verified` and
  `negative_reviewed` are ground truth; `machine_proposed` is a model
  prediction and `unreviewed` is an unknown.
* The class schema is versioned (`class_schema_version`). The current
  schema is **single-class**: `0 = person`.

### 2.5 Compatibility rules

* Manifests carry `contract_version`, `manifest_version`,
  `class_schema_version` and `dataset_version_id`. A manifest written by
  an older contract is still readable; the version fields say which rules
  produced it.
* The catalog migrates forward only (`PRAGMA user_version`), and a
  migration never rewrites historical rows.
* A dataset version used by a completed experiment stays identifiable:
  the experiment row stores both the `dataset_version_id` and the
  `manifest_sha256`, and the manifest file is never rewritten in place —
  a new version is a new file.

---

## 3. Dataset layout and annotation format

```
datasets/people_v1/
├── data.yaml               # YOLOv5 entry point (generated)
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

`data.yaml` follows the vendored YOLOv5 convention exactly:

```yaml
path: ../datasets/people_v1     # relative to the vendored yolov5/ root
train: images/train
val: images/val
test: images/test
names:
  0: person
```

`path:` is relative to `yolov5/` because that is where
`yolov5/utils/general.check_dataset` resolves a relative path from — the
same convention `yolov5/data/coco128.yaml` uses. `scripts/build_dataset.py`
computes it with `os.path.relpath(root, yolov5_root)`.

### 3.1 Group folders and the mirrored label layout

A dataset is expected to keep the frames of one recording session in a
sub-directory, because that sub-directory name is the **source group**
the splitter uses to keep a session inside one split:

```
datasets/people_v1/
├── images/train/session_2026_10_20_a/frame_0001.jpg
├── images/val/session_2026_10_20_b/frame_0001.jpg
├── labels/train/session_2026_10_20_a/frame_0001.txt
└── labels/val/session_2026_10_20_b/frame_0001.txt
```

The path inside `images/<split>/` is mirrored into `labels/<split>/`, so
`images/train/session_a/f.jpg` pairs with `labels/train/session_a/f.txt`.
Both the validator and the training preflight walk these trees
**recursively**; a frame nested one level deeper still finds its label.

Images that sit directly in `images/<split>/` (no group folder) have no
source group and are reported as orphans (§4.2).

### 3.2 Annotation format

`<class_id> <x_center> <y_center> <width> <height>`, all coordinates
normalised to `[0, 1]`, box described by centre and size.

### 3.3 Validation rules

`ml/validate.py` checks and reports (never repairs):

| Code | Level | Meaning |
|---|---|---|
| `missing_label` | error | image without a label file |
| `missing_image` | error | label file without an image |
| `label_malformed_row` | error | a row that does not have exactly 5 fields |
| `label_non_numeric` | error | a row with a non-numeric field |
| `label_non_integer_class` | error | a fractional class id |
| `label_class_out_of_range` | error | class id not in the class schema |
| `label_coords_out_of_bounds` | error | a coordinate outside `[0, 1]` |
| `label_zero_area_box` | error | zero or implausible area |
| `label_box_outside_image` | error | the box extends past the image edge |
| `image_unreadable` | error | the file is corrupt or not an image |
| `image_unsupported_format` | error | not a format YOLOv5 can read |
| `duplicate_image` | error | byte-identical to another image |
| `empty_label_unclassified` | warning | empty label file with no declared status |
| `empty_label_reviewed_negative` | info | empty label file declared as a reviewed negative |
| `near_duplicate_image` | warning | perceptual-hash distance ≤ 4 from another image |
| `inconsistent_dimensions` | warning | more than one image size in the dataset |
| `image_too_small` | warning | a side below 32 px |
| `image_unusual_mode` | info | colour mode outside RGB/L/P |
| `split_images_dir_missing` | warning | the split has no images directory |
| `split_empty` | warning | the split has no images |

**Empty label files are valid.** An empty label file means "reviewed, no
person present". It is only *treated* as a reviewed negative when the
manifest declares `annotation_status = negative_reviewed`; otherwise it is
a warning so that a human decides. This mirrors YOLOv5's own treatment of
empty labels as backgrounds (`yolov5/utils/dataloaders.py`).

Reports are emitted as JSON (`counts`, `summary`, `findings`) and as a
human-readable text summary. `scripts/validate_dataset.py --preview-out`
additionally renders annotated copies of the images for visual review
(`ml/preview.py`).

---

## 4. Leakage-safe splitting

`ml/split.py`.

### 4.1 Group-first assignment

Frames from the same video or recording session share a `source_group`
and are **always** placed in the same split. The group is the directory
name under the split directory:

```
images/train/session_2026_10_20_a/frame_0001.jpg   -> group "session_2026_10_20_a"
```

The algorithm is a pure function of `(seed, ratios, group set)`:

1. groups are sorted by id, then shuffled with `random.Random(seed)`;
2. each group is assigned to the split with the largest remaining deficit
   (target − assigned), ties broken by the fixed order train → val → test.

Re-running with the same inputs reproduces the same split exactly, which
is what makes an experiment reproducible. This is asserted by the test
`test_split_is_deterministic_for_a_seed`.

### 4.2 Orphan images

Images with no `source_group` are reported. The default policy
(`orphan_policy = singleton`) treats each as its own group and raises a
**warning** that states plainly that their split membership cannot be
guaranteed leakage-free. `orphan_policy = error` refuses the split
instead.

### 4.3 Leakage checks

After assignment, three checks run over the whole dataset:

* `group_leakage` (**error**) — a source group spanning two splits.
* `duplicate_leakage` (**error**) — byte-identical content in two splits.
* `near_duplicate_leakage` (**warning**) — perceptual hash distance ≤ 4
  (configurable) between images in two splits.

An error-level leakage finding **stops** `build_dataset.py` before a
manifest is written.

### 4.4 The held-out test set

The `test` split is the held-out set used for the final baseline-vs-candidate
comparison. The report records the dataset version id and the split name
for every evaluation, and `ml/evaluate.compare()` refuses to treat two
evaluations as comparable when they used different dataset versions or
different splits. The test set is not used for threshold or
hyper-parameter selection.

---

## 5. Fine-tuning data strategy

### 5.1 What additional data can and cannot fix

The current checkpoint is the **stock COCO YOLOv5s**. It has never seen
airport X-ray queues, and the failure modes below are therefore
*candidate* categories, not verified claims about the model.

| Category | What data would help | Typical cause |
|---|---|---|
| `crowd_dense` | many overlapping people | COCO's average person density is far below a screening queue |
| `small_person` | distant people | COCO images are mostly near-field |
| `truncated_body` | bodies cut off by the frame edge | queue people are often partially out of frame |
| `low_confidence` | ambiguous poses, low light | COCO is mostly well-lit outdoor imagery |
| `no_detection` | scene-level misses | overhead camera angle, X-ray machine in view |

The miner flags these categories automatically (§6), but a category is
only a **hypothesis about where the model struggles**. It becomes a
finding only after a human confirms it.

### 5.2 Hard-example collection workflow

`ml/collect.py` + `scripts/collect_hard_examples.py`:

1. run the **current** checkpoint over permitted source material;
2. classify each frame against the categories above;
3. write the candidates to a CSV, a `run.json` and a `REVIEW.md` sheet;
4. a human reviews the candidates and produces corrected labels;
5. only the corrected, reviewed frames are added to a new dataset version;
6. the new version is re-validated and re-split (the old one stays intact).

**Machine output is never promoted to ground truth.** Every candidate
carries `annotation_status = machine_proposed` and a review note. There
is no code path in this toolkit that writes a model prediction into a
label file.

### 5.3 Privacy and authorisation

Frames from a camera contain identifiable people. The miner therefore
requires **two** independent flags before it writes a single pixel:

* `--retain-frames`, and
* `--consent-acknowledged`.

Without both, only derived statistics (counts, confidence values, box
areas) are written — no images. The notice printed before every run
states the obligations: the source must be permitted for this use, the
people in the frames must be lawfully processable, and the retention
window (default 30 days) must be acceptable. The collected directory is
git-ignored for the same reason.

---

## 6. The metadata catalog

`ml/catalog.py` → `datasets/metadata/dataset_catalog.db` (SQLite,
stdlib `sqlite3`, **not** the runtime monitoring database).

### 6.1 Creating the catalog

The catalog is created **explicitly**, never as a side effect of an
inspection command -- an auto-created catalog would look like a populated
one:

```bash
venv/bin/python scripts/catalog.py init
# [OK] catalog ready: datasets/metadata/dataset_catalog.db
# [INFO] schema version: 1
```

`init` is idempotent: it creates the parent directory if needed, applies
any pending migrations, and leaves existing records untouched. A dataset
build also creates the catalog if it is missing.

The inspection commands (`stats`, `datasets`, `experiments`, `artifacts`,
`metrics`) report a missing catalog clearly instead of failing obscurely:

```
$ venv/bin/python scripts/catalog.py stats
[ERROR] catalog not found: datasets/metadata/dataset_catalog.db
The dataset catalog does not exist yet.
  Create it with:  venv/bin/python scripts/catalog.py init
  (a dataset build also creates it automatically)
```

They exit with code **3** in that case, and `--json` returns
`{"exists": false, "error": "catalog_not_initialized", "hint": ...}`.

### 6.2 Why a database at all

The **manifest** is the source of truth and is sufficient to reproduce a
dataset on its own. The catalog exists because experiments need to be
*looked up*: "which dataset version, which seed, which base checkpoint,
which metrics" — without scanning the filesystem. It indexes manifests; it
does not replace them.

The catalog never stores image binaries or video files, only paths,
hashes and metadata.

### 6.3 Schema (v1, `PRAGMA user_version = 1`)

| Table | Purpose |
|---|---|
| `dataset_version` | one row per dataset version: manifest path + hash, seed, split policy |
| `source_asset` | raw sources: kind, ref, hash, group, licence, restrictions |
| `image` | per-image metadata: relpath, hash, dimensions, group, annotation status, phash |
| `split_assignment` | image → split, with the group it was assigned by |
| `validation_report` | a stored report: hash, status, counts, findings |
| `experiment` | resolved config + config hash, dataset version, base checkpoint + hash, environment, status |
| `model_artifact` | checkpoints: role (`best` / `last`), path, sha256, size |
| `metric` | one evaluation metric per (experiment, metric, split) |

Constraints: `PRAGMA foreign_keys = ON`, `ON DELETE CASCADE` from dataset
version to images and splits, `UNIQUE (name, manifest_sha256)` on dataset
versions, `UNIQUE (experiment_id, role)` on artifacts. Indexes on
`image(sha256)`, `image(source_group)`, `split_assignment(split)` and
`metric(experiment_id)`.

### 6.4 Idempotency and immutability

Registration uses `INSERT … ON CONFLICT DO UPDATE` inside a transaction,
so registering the same dataset version or experiment twice is safe.

**A dataset is never *declared* immutable.** `Catalog.verify_dataset()`
recomputes the SHA-256 of every referenced file and reports drift;
`ml/manifest.verify_manifest()` does the same for the manifest body
itself. Immutability is a verified property, not a naming convention.

### 6.5 API

The API is deliberately small: `register_dataset_version`,
`register_images`, `register_splits`, `register_validation_report`,
`register_experiment`, `register_artifact`, `record_metric`,
`list_dataset_versions`, `list_images`, `list_experiments`,
`list_artifacts`, `list_metrics`, `verify_dataset`, `stats`.

---

## 7. Training and experiment reproducibility

`ml/experiment.py` + `scripts/train_person.py`.

Every run records:

| Field | Source |
|---|---|
| `experiment_id`, timestamps | generated |
| dataset version id + `manifest_sha256` | passed in / read from the manifest |
| base checkpoint path + SHA-256 | hashed at launch |
| resolved config + its SHA-256 | the `ExperimentConfig` |
| image size, batch size, epochs, optimizer, seed, workers, hyp file, augmentation flags | the resolved config |
| class mapping and model configuration | the dataset's `data.yaml` (`names`, `nc`) |
| Python, PyTorch, torchvision, CUDA, MPS, ultralytics, YOLOv5 revision, platform | `capture_environment()` |
| the exact command | `ExperimentConfig.build_command()` |
| log path and stdout | the subprocess log |
| `best.pt` and `last.pt` paths + hashes | `find_checkpoints()` + `sha256_file` |
| failures, warnings, incomplete runs | `TrainingRun.status` / `.error` |

The **resolved** configuration is stored, not the shell command, so a run
can be reproduced even if the CLI changes. The base checkpoint is only
ever read; the vendored trainer writes to a separate run directory, and
`models/yolov5s.pt` is never modified — asserted by
`test_original_checkpoint_is_not_modified`.

### 7.1 The training preflight

`yolov5/train.py` will happily train on a dataset with no images, with
missing label files or with out-of-range class ids — it simply produces a
useless checkpoint. `scripts/train_person.py` therefore runs a **preflight**
before the trainer is invoked (`ml/preflight.py`) and refuses the run with
exit code **4** if any fatal check fails:

| Check | Fatal | Meaning |
|---|---|---|
| `data_yaml` | yes | the dataset YAML exists and parses |
| `data_yaml_names` | yes | the class schema is declared |
| `dataset_root` | yes | the dataset root resolves to a real directory |
| `train_images` / `val_images` | yes | the split contains images |
| `train_labels` / `val_labels` | yes | every image has a label file and every label row is valid (class id in the schema, normalised coordinates in range, non-zero area) |
| `manifest` | yes | a dataset manifest exists for this root, its status is `ready`, and its recorded hashes still match the files on disk |

`--dry-run` runs the preflight and prints the resolved command without
training, which is the quickest way to check a dataset:

```bash
venv/bin/python scripts/train_person.py --data datasets/people_v1/data.yaml \
    --weights models/yolov5s.pt --dry-run
```

`--skip-preflight` bypasses the gate. It is recorded in the experiment
metadata and in `experiment.json`, so a run that skipped the checks stays
visible as such.

### 7.2 Using the vendored trainer

The training command targets `yolov5/train.py` directly. The modern
`ultralytics` CLI is **not** used and is not assumed to be compatible.
A test (`test_command_contains_only_flags_the_vendored_trainer_accepts`)
parses `yolov5/train.py` and asserts that every flag the launcher emits
is actually accepted — this caught a `--project-name` flag that does not
exist in this revision.

---

## 8. Evaluation and quality gates

`ml/evaluate.py` + `scripts/evaluate_model.py`.

### 8.1 Metric definitions

Metrics come from the vendored `yolov5/val.py`, so they follow YOLOv5's
definitions:

* **precision / recall** — at the validation confidence threshold, with
  the IoU 0.50 matching used by the COCO-style evaluator.
* **mAP@0.5** and **mAP@0.5:0.95** — averaged over the 101-point
  interpolation in `yolov5/utils/metrics.py`.
* **per-class** — the same quantities restricted to one class.

`ml/evaluate.py` parses the summary line of `val.py` and the
`results.csv` of a training run (selecting the row with the highest
YOLOv5 fitness, `0.1·mAP50 + 0.9·mAP5095`, which is the checkpoint
YOLOv5 itself keeps as `best.pt`).

**`--single-cls` is required.** The dataset schema is single-class, so
the validator is invoked with `--single-cls`. Without it YOLOv5 asserts
that the checkpoint's class count matches the dataset and refuses to
run — the stock 80-class COCO baseline cannot be evaluated against a
1-class dataset. With `--single-cls`, YOLOv5 maps every prediction to
class 0 and uses class-agnostic NMS, which is exactly the comparison
wanted against a `person`-only dataset. The flag is on by default in
`scripts/evaluate_model.py`; `--no-single-cls` disables it.

### 8.2 Comparison rules

* The candidate and the baseline must be evaluated on the **same dataset
  version and the same split**; `compare()` marks the comparison
  *not comparable* otherwise, and the verdict is REJECT.
* Deltas are reported per metric. A comparison states the absolute values
  and the deltas, never a bare "improved".
* Training loss alone is never used as evidence of improvement.

### 8.3 Acceptance criteria (provisional)

`DEFAULT_ACCEPTANCE_CRITERIA` is flagged `provisional: true` and is
printed as such in every report:

| Metric | Rule | Meaning |
|---|---|---|
| `map_50` | ≥ baseline | mAP@0.5 must not fall below the baseline |
| `map_50_95` | ≥ baseline | mAP@0.5:0.95 must not fall below the baseline |
| `map_50` | ≥ baseline + 0.02 | target improvement |
| `recall` | ≥ baseline − 0.01 | recall may not regress (missed people are the operationally costly error) |

These are **engineering defaults, not an agreed product requirement**.
They must be ratified before they gate a release.

### 8.4 Latency and throughput

`yolov5/val.py --task speed` provides YOLOv5's own latency benchmark.
Inference latency for the runtime path is a separate concern measured by
the runtime's own instrumentation (`SystemMetrics.inference_ms`); this
workstream does not duplicate it.

---

## 9. Commands

```bash
# create the metadata catalog (idempotent, safe to re-run)
venv/bin/python scripts/catalog.py init

# validate what is in a dataset right now
venv/bin/python scripts/validate_dataset.py --root datasets/people_v1 \
    --report datasets/reports/people_v1.json

# render annotation previews for human review
venv/bin/python scripts/validate_dataset.py --root datasets/people_v1 \
    --preview-out datasets/reports/preview

# build a dataset version: validate -> split -> manifest -> catalog
venv/bin/python scripts/build_dataset.py --root datasets/people_v1 \
    --name people_v1 --seed 20261020 --ratios 0.7/0.2/0.1

# inspect the catalog
venv/bin/python scripts/catalog.py stats
venv/bin/python scripts/catalog.py datasets
venv/bin/python scripts/catalog.py experiments
venv/bin/python scripts/catalog.py verify --dataset-version dsv_xxxxxxxx

# mine hard examples (machine proposals only)
venv/bin/python scripts/collect_hard_examples.py --source input/clip.mp4 \
    --weights models/yolov5s.pt --out datasets/collected/run_001

# fine-tune, with full experiment provenance
venv/bin/python scripts/train_person.py --data datasets/people_v1/data.yaml \
    --weights models/yolov5s.pt --epochs 50 --imgsz 640 --batch 8 --device mps

# evaluate against the baseline
venv/bin/python scripts/evaluate_model.py --weights runs/train/people/weights/best.pt \
    --baseline models/yolov5s.pt --data datasets/people_v1/data.yaml --split test

# make targets
make catalog-init
make dataset-validate
make dataset-build
make dataset-check
make catalog-stats
make ml-help
```

### 9.1 Exit codes

| Command | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| `validate_dataset.py` | no errors | error-level findings | bad usage | — | — |
| `build_dataset.py` | version built & registered | validation / leakage / registration failure | bad usage | dataset contains no images | — |
| `catalog.py` | command succeeded | verification failed | bad usage | catalog not initialised | — |
| `train_person.py` | training completed | training failed or no checkpoint | bad usage | — | preflight refused the run |
| `evaluate_model.py` | evaluation completed | evaluation failed | bad usage | — | — |

An **empty dataset is refused** by `build_dataset.py` with exit code 3:
registering a version that contains no images would create a record that
looks usable but cannot train anything. `--allow-empty` registers it
explicitly instead, with `status = "empty"` in the manifest and in the
catalog, and the training preflight refuses to train on it.

### 9.2 Synthetic fixture (plumbing only)

`scripts/make_synthetic_fixture.py` generates a **synthetic** dataset of
procedurally drawn shapes. It exists so the training plumbing can be
smoke-tested without any real imagery. The images have **no detection
value whatsoever** and the script **refuses to write inside the
repository's `datasets/` directory**, so synthetic data can never be
mistaken for reviewed data:

```bash
venv/bin/python scripts/make_synthetic_fixture.py --out /tmp/ml-fixture \
    --images 24 --seed 7
```

---

## 10. Tests

`tests/test_dataset_tooling.py` (69 tests) and `tests/test_dataset_cli.py`
(36 tests), written as `unittest.TestCase` classes so both runners in this
project collect them (`make test` → `unittest discover`; `pytest tests/`).
The CLI tests drive the real command-line entry points in-process, inside
temporary directories, and assert their exit codes.

| Area | Covered |
|---|---|
| Contract | canonical JSON, hash stability, content-addressed ids, class-schema validation, ground-truth flags |
| Annotation parsing | valid rows, class id out of range, out-of-bounds coordinates, zero-area boxes, malformed rows, non-numeric rows, non-integer class, box outside the image |
| Dataset validation | missing labels, missing images, corrupt images, empty labels (declared and undeclared), duplicate images, inconsistent dimensions, report serialisation, perceptual-hash stability |
| Splitting | determinism, seed sensitivity, group containment, group/duplicate/near-duplicate leakage, orphan handling, full assignment |
| Manifest | hash stability, tamper detection, modified-file detection, round trip, broken manifest |
| Catalog | fresh schema, idempotent re-open, idempotent registration, image/split/experiment/artifact/metric registration, hash verification, separation from the runtime database |
| Experiments | config hash stability, JSON serialisation, command construction, flag compatibility with the vendored trainer, checkpoint hashing, incomplete-run recording, original checkpoint preservation |
| Evaluation | `results.csv` parsing, stdout parsing, report serialisation, delta computation, regression rejection, non-comparable evaluations, provisional-criteria flag |
| Catalog lifecycle | `init` creates the file and its parent directory; `init` is idempotent; `init` preserves existing records; migrations are forward-only; inspection on a missing catalog returns exit 3 with an actionable message and a JSON form; inspection on an initialised catalog succeeds; `verify` reports an unknown version |
| Build | a valid dataset builds and registers version + images + splits; the catalog is created when missing; rebuilding three times creates no duplicates; an empty dataset is refused with exit 3; `--allow-empty` registers an explicitly empty version; a failed validation registers no version; a failed registration leaves nothing behind; a failed write inside the transaction rolls the version row back; a missing root is a usage error |
| Preflight | passes for a valid built dataset; fails on an empty dataset, a missing label file, an invalid label, a missing manifest, a manifest that no longer matches the files, and an `empty` manifest status; training is refused with exit 4 and no run directory is created; the result is serialisable |
| CLI exit codes | `validate_dataset` 0/1/2; `build_dataset` 0/1/2/3; `catalog` 0/1/2/3; `train_person` 2/4 |

All tests use temporary directories and synthetic fixtures. **No test
downloads anything, touches `models/`, or writes to `datasets/`.**

### 10.1 End-to-end verification that was actually run

Against a scratch copy of the repository (the repository itself was never
modified during development):

| Step | Command | Result |
|---|---|---|
| Baseline suite | `unittest discover -s tests` | 137 → **242 passed** (+105) |
| Baseline suite | `pytest tests/` | 169 → **274 passed** (+105) |
| Synthetic fixture | `scripts/make_synthetic_fixture.py --out /tmp/ml-fixture --images 24 --seed 7` | 24 images, 14/5/5 across train/val/test |
| Validation | `scripts/validate_dataset.py --root /tmp/ml-fixture` | PASS, 0 errors |
| Version build | `scripts/build_dataset.py --root /tmp/ml-fixture --name fixture_v1 --seed 20261020` | manifest + `data.yaml` + catalog rows written |
| Hash verification | `scripts/catalog.py verify --dataset-version dsv_…` | **OK** — every recorded hash matches the file on disk |
| Manifest verification | `verify_manifest(..., check_files=True)` | **OK** |
| **Training smoke** | `scripts/train_person.py --data /tmp/ml-fixture/data.yaml --weights models/yolov5s.pt --epochs 1 --imgsz 160 --batch-size 4 --device mps` | **completed**, 1 epoch on MPS, `best.pt` + `last.pt` written (14.2 MB each) |
| Provenance | catalog `experiment` row | id, config hash, base-checkpoint hash, dataset version, device, torch/ultralytics/MPS/CUDA versions, YOLOv5 revision `20d1d78a…`, log path, command — all recorded |
| Artifacts | catalog `model_artifact` rows | `best` and `last` recorded with their own SHA-256 |
| Evaluation | `scripts/evaluate_model.py --weights …/best.pt --baseline models/yolov5s.pt --split test` | both sides evaluated; deltas and the acceptance verdict produced |
| Privacy control | `collect_hard_examples.py` without `--consent-acknowledged` | **0 frames retained**, no `frames/` directory created |
| Privacy control | `collect_hard_examples.py --retain-frames --consent-acknowledged` | 14 frames retained, `candidates.csv` + `REVIEW.md` written |
| Original checkpoint | `shasum -a 256 models/yolov5s.pt` | `8b3b748c1e59…` — **unchanged** after the training run |
| Vendored YOLOv5 | `diff -r --exclude=.git` | **identical** — the upstream checkout was not modified |
| Catalog init | `scripts/catalog.py init` (twice) | **OK**, schema version 1, idempotent |
| Catalog on a missing file | `scripts/catalog.py stats` | clear message + exit code **3**, `--json` form reports `catalog_not_initialized` |
| Empty dataset build | `scripts/build_dataset.py --root datasets/people_v1 --name people_v1` | **refused**, exit code **3**, no dataset version registered, validation report kept |
| Empty dataset, explicit | `… --allow-empty` | registered with `status = "empty"`; the preflight refuses it |
| Build idempotency | the same build three times | 1 dataset version, 1 validation report, no duplicates |
| Registration rollback | a failing image insert | the version row is rolled back with it — 0 versions, 0 images |
| Training preflight | `scripts/train_person.py --data <empty> …` | **refused**, exit code **4**, no run directory created |
| Training preflight | `scripts/train_person.py --data <valid> … --dry-run` | **PASS** on every check, then the resolved command is printed |

**The training smoke test is a plumbing test, not a fine-tuning result.**
It ran on procedurally drawn shapes, so the metrics it produced
(mAP@0.5 ≈ 0.003) are meaningless as detection quality and are not
reported anywhere as a model-quality claim. The only thing it
demonstrates is that the command line, the dataset description, the
checkpoint production, the provenance capture and the evaluation path all
work end to end on this machine.

---

## 11. Known limitations and prerequisites

1. **No dataset exists.** `input/` is empty and there is no annotated
   imagery anywhere in the repository. The infrastructure is complete and
   tested; the data has to be supplied. The tooling was therefore
   verified against **synthetic fixtures** only.
2. **No training has been performed on real data**, and no metric in this
   document is a measurement of a fine-tuned model. The only training that
   can be run on this machine is a smoke test on synthetic data, and it
   proves plumbing, not accuracy.
3. **No CUDA.** Real fine-tuning here is MPS/CPU only — orders of
   magnitude slower than a datacentre GPU. The launcher supports
   `--device 0` for the training host.
4. **`Arial.ttf` is missing** from `yolov5/`; the first `train.py`/`val.py`
   run downloads it. The dataset tooling itself is fully offline.
5. **The vendored YOLOv5 is not modified** by this workstream. It is an
   upstream clone with its own contribution rules.
6. **Near-duplicate detection is perceptual, not semantic.** The 64-bit
   dHash catches re-encodings and light edits; it does not catch a
   re-shot, differently-cropped version of the same scene. Those need a
   human. It also degenerates on near-flat images (e.g. synthetic
   renders), where many different images share a hash — the finding is a
   warning, not an error, precisely because of this.
7. **Per-frame event data is out of scope.** Passenger enter/exit events
   are a runtime concern; the dataset stores what detection training
   needs.
8. **Group-level splitting needs enough groups.** The split is assigned per
   source group, so a dataset with only a handful of groups cannot hit
   the requested ratios (three groups of three images yield 6/3/0 across
   train/val/test). Real datasets have one group per recording session
   and are fine; small fixtures should not be read as a ratio failure.
9. **`data.yaml` is regenerated by the build.** The `path:` entry is
   relative to the vendored `yolov5/` root, which is where
   `yolov5/utils/general.check_dataset` resolves it from. Moving the
   repository therefore requires re-running the build.

---

## 12. Extension points

| Extension | Where |
|---|---|
| More classes (e.g. `bag`, `staff`) | add to `DEFAULT_CLASS_SCHEMA` in `ml/contract.py`, bump `CLASS_SCHEMA_VERSION`, add a migration to the catalog |
| COCO-format annotations | add a parser next to `parse_label_file` in `ml/validate.py` |
| Semantic segmentation labels | YOLOv5 supports segment labels (7-column rows); the parser already detects them and would need a segment-aware validator |
| Active learning | `ml/collect.py` already emits a priority-ordered review sheet |
| DVC / external dataset storage | the manifest carries hashes, so a dataset can live outside the repository and still be verified with `verify_manifest` |
| A second dataset version | a new manifest file; the catalog keeps both |

---

*Last verified against the code: 2026-10-20.*
