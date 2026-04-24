"""
create_T_120_splits.py
======================
Standalone script that produces data_splits_T_120_sampled.json --
a pre-ictal-only training manifest for the ablation study.

Run ONCE on the cluster before any tuning or training job. Reads the
authoritative sources (data_splits.json, EDF headers, seizure annotation
.xlsx files) and produces a manifest without creating, moving, or
deleting any .npy files on disk. The original non_seizure/ folders are
preserved intact for future self-supervised learning (SSL) work.

Motivation
----------
This manifest restricts the non-ictal training class to a narrow
pre-ictal window placed strictly before each seizure. For every
annotated seizure starting at time T (seconds from recording start),
only non-ictal segments whose boundaries satisfy

        seg_start >= T - 120  AND  seg_end <= T - 60

(full containment in the interval [T - 120 s, T - 60 s]) are retained.
This defines a 60-second pre-ictal window separated from seizure
onset by a 60-second lead (the "seizure prediction horizon").

Comparison against the peri-ictal manifest
------------------------------------------
This manifest is intended for head-to-head comparison against the
earlier peri-ictal manifest (data_splits_nonictal_sampled.json, produced
historically by create_balanced_splits.py), which retained all non-ictal
segments within +/-60 s of seizure boundaries (both pre- and post-ictal)
plus a 5% random subsample of distant background. That manifest was
designed for seizure DETECTION and produced a 1:2.37 ictal-to-non-ictal
ratio. The present manifest is designed for a different question:
whether a classifier can discriminate ictal activity from EEG in the
minute preceding seizure onset (separated by a 60-second lead).

Both manifests share the SAME data_splits.json input, the SAME
exclusion criteria (m254 removal, amplitude filter, NaN/Inf filter),
the SAME validation partition (pass-through), the SAME random seed
(42), and the SAME .npy file universe on disk. Only the non-ictal
selection rule differs, so downstream differences in model performance
are attributable to the selection rule alone.

Scientific rationale
--------------------
Restricting negatives to a fixed pre-ictal window before seizure onset
follows the seizure-prediction framework articulated by Maiwald et al.
(2004), in which a "seizure prediction horizon" (SPH) separates
predicted-state evidence from ictal onset and a "seizure occurrence
period" (SOP) defines the interval during which the predicted event
must occur. Here SPH = 60 s and SOP = 60 s.

A 60-second lead was chosen to (i) exclude immediate peri-ictal
spiking and EEG transients that blur the boundary with ictal activity
(Litt et al., 2001), (ii) accommodate the approximate onset-annotation
resolution reported for rodent EEG (Luttjohann et al., 2009), and
(iii) lie within the range of pre-ictal intervals where distinctive
EEG dynamics have been reported in intracranial and scalp recordings
(Mormann et al., 2007).

Class imbalance handling
------------------------
Under this strategy the non-ictal class is NOT broader than the ictal
class. The per-seizure ceiling on retained non-ictal segments is
approximately (PREICTAL_WIDTH_SEC - WIN_LEN/FS) / (STEP/FS) + 1 ~= 23
segments for a 5 s window with a 2.5 s step. Across N_sz annotated
seizures the non-ictal count is bounded above by ~23 * N_sz. Because
ictal segments are kept in full, the ictal-to-non-ictal ratio will be
approximately 1:1 or ictal-majority, so NO additional class-reweighting
is required: the offline selection rule itself delivers an approximately
balanced corpus. Downstream training scripts retain
BCEWithLogitsLoss(pos_weight=1.0), unchanged from the peri-ictal
manifest.

This is consistent with Shoeb & Guttag (2010) and more recent seizure-
prediction pipelines (Kuhlmann et al., 2018; Rasheed et al., 2020),
which train binary classifiers on fixed-size pre-ictal and ictal
windows without additional reweighting, relying on the selection rule
itself to deliver class balance.

Window definition
-----------------
For a seizure at onset time T (seconds from recording start):
    Window A = [T - 120, T - 60]
A non-ictal segment with boundaries [seg_start, seg_end] is retained
iff seg_start >= T - 120 AND seg_end <= T - 60 for at least one
annotated seizure T in the same recording (full containment).

Post-ictal segments, inter-ictal (>120 s from any seizure) segments,
and segments within the 60 s immediately preceding seizure onset
(T - 60 to T) are all DROPPED.

Seizures whose onset is <120 s into the recording have no usable
Window A and are skipped (their ictal segments are still kept via
the normal ictal pass-through).

Downsampling strategy
---------------------
1. Remove m254 (ictal-free subject -- cannot participate in pre-ictal
   sampling).
2. Remove extreme segments (max|x| > 1000 or non-finite values --
   indicating failed z-score normalisation from interrupted
   preprocessing).
3. Keep ALL ictal segments.
4. Keep non-ictal segments fully contained in Window A of any seizure.
5. Drop all other non-ictal segments (post-ictal, inter-ictal,
   immediate pre-ictal <60 s).
6. Val partition passed through unchanged.

Time-position reconstruction
-----------------------------
The filename convention {mouse_id}_{ictal|nonictal}_{index:05d}.npy
uses a sequential per-class counter, NOT a grid position. To recover
each segment's temporal position we re-read each mouse's EDF header
(preload=False -- reads only metadata, ~10 ms per file) and annotation
file, rebuild the full segment grid, and map the k-th non-ictal grid
position to nonictal_{k:05d}.npy.

Output
------
/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_T_120_sampled.json

Pipeline position
-----------------
1. generate_data_splits.py         -> data_splits.json
2. create_T_120_splits.py          -> data_splits_T_120_sampled.json  (this script)
3. tcn_HPT_binary.py / tune_*.py   -> tuning (loads pre-ictal manifest)
4. TCN.py / MultiScaleTCN.py etc   -> training (loads pre-ictal manifest)

References
----------
Maiwald, T., Winterhalder, M., Aschenbrenner-Scheibe, R., Voss, H. U.,
    Schulze-Bonhage, A., & Timmer, J. (2004). Comparison of three
    nonlinear seizure prediction methods by means of the seizure
    prediction characteristic. Physica D, 194(3-4), 357-368.
Mormann, F., Andrzejak, R. G., Elger, C. E., & Lehnertz, K. (2007).
    Seizure prediction: the long and winding road. Brain, 130(2),
    314-333.
Freestone, D. R., Karoly, P. J., & Cook, M. J. (2017). A forward-
    looking review of seizure prediction. Current Opinion in
    Neurology, 30(2), 167-173.
Kuhlmann, L., Lehnertz, K., Richardson, M. P., Schelter, B., &
    Zaveri, H. P. (2018). Seizure prediction -- ready for a new era.
    Nature Reviews Neurology, 14(10), 618-630.
Rasheed, K., Qayyum, A., Qadir, J., Sivathamboo, S., Kwan, P.,
    Kuhlmann, L., O'Brien, T., & Razi, A. (2020). Machine learning
    for predicting epileptic seizures using EEG signals: A review.
    IEEE Reviews in Biomedical Engineering, 14, 139-155.
Shoeb, A. & Guttag, J. (2010). Application of Machine Learning to
    Epileptic Seizure Detection. Proceedings of the 27th International
    Conference on Machine Learning (ICML).
Litt, B., Esteller, R., Echauz, J., D'Alessandro, M., Shor, R.,
    Henry, T., Pennell, P., Epstein, C., Bakay, R., Dichter, M., &
    Vachtsevanos, G. (2001). Epileptic seizures may begin hours in
    advance of clinical onset: a report of five patients. Neuron,
    30(1), 51-64.
Luttjohann, A., Fabene, P. F., & van Luijtelaar, G. (2009). A revised
    Racine's scale for PTZ-induced seizures in rats. Physiology &
    Behavior, 98(5), 579-586.

Usage
-----
python create_T_120_splits.py

RESEARCH REPORTING NOTE
-----------------------
Report the following in the Methods section:

  "Non-ictal training segments were selected using a fixed pre-ictal
   window strategy. For each annotated seizure with onset T, only
   non-ictal segments whose boundaries satisfied
   [T - 120 s, T - 60 s] (full containment) were retained. A 60-second
   lead separated the pre-ictal window from seizure onset to exclude
   immediate peri-ictal transients (Maiwald et al., 2004; Litt et al.,
   2001). Post-ictal, inter-ictal, and immediate pre-ictal (<60 s)
   segments were excluded entirely. The resulting corpus contained
   N_ictal ictal and N_nonictal non-ictal segments (ratio X:1);
   because the selection rule itself produces approximate class
   balance, no additional class-reweighting was applied
   (pos_weight = 1.0). The manifest was generated once offline and
   reused across all tuning and training runs. Seed = 42. Subject m254
   (no ictal segments) was excluded. Segments with max|x| > 1000 or
   non-finite values (indicating incomplete z-score normalisation)
   were removed."

Parameters to report:
  PREICTAL_LEAD_SEC    : 60 s   (gap between window end and seizure onset)
  PREICTAL_WIDTH_SEC   : 60 s   (pre-ictal window duration)
  AMPLITUDE_THRESHOLD  : 1000.0
  SEED                 : 42
  Excluded subjects    : m254
"""

import json
import logging
import sys
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import mne
import pandas as pd
import matplotlib
matplotlib.use("Agg")                               # non-interactive backend for cluster
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42                         # reproducibility seed (kept for parity with peri-ictal manifest)
FS = 500                          # EEG sampling rate in Hz (must match preprocessing)
WIN_LEN = 2500                    # segment length in samples: 5 s * 500 Hz
STEP = 1250                       # step size in samples: 2.5 s (50% overlap)
PREICTAL_LEAD_SEC = 60.0          # gap between pre-ictal window end and seizure onset
PREICTAL_WIDTH_SEC = 60.0         # pre-ictal window duration (Window A width)
AMPLITUDE_THRESHOLD = 1000.0      # max|x| filter (failed z-score normalisation)

# -- Paths -----------------------------------------------------------------
# Input: original data_splits.json produced by generate_data_splits.py
SPLITS_PATH = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json")
# Output: pre-ictal manifest consumed by all tuning and training scripts
OUTPUT_PATH = Path("/home/people/22206468/scratch/INPUT_T_120/data_splits_outputs/data_splits_T_120_sampled.json")
# Seizure annotation Excel files (one per mouse: {mouse_id}_xlsx.xlsx)
ANNOTATION_DIR = Path("/home/people/22206468/scratch/seizure_times_updated")

# EDF directories -- one per preprocessing batch. Each contains the raw
# .edf recordings for a subset of mice. We only read headers (preload=False)
# to get n_samples for grid reconstruction, NOT the signal data.
EDF_DIRS = [
    Path("/home/people/22206468/scratch/EEG_TRAINING"),       # preprocessing_binary.py output
    Path("/home/people/22206468/scratch/EEG_TRAINING_2"),     # preprocessing_binary_train_2.py output
    Path("/home/people/22206468/scratch/EEG_TRAINING_3"),     # preprocessing_binary_train_3.py output
    Path("/home/people/22206468/scratch/EEG_TRAINING_4"),     # preprocessing_binary_train_4.py output
    Path("/home/people/22206468/scratch/EEG_TRAINING_5"),     # preprocessing_binary_train_5.py output
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("T_120_splits")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)

fh = logging.FileHandler(OUTPUT_PATH.parent / "create_T_120_splits.log",
                          mode="a", encoding="utf-8")
fh.setFormatter(fmt)
logger.addHandler(fh)


# ---------------------------------------------------------------------------
# Helpers (reused from preprocessing_binary.py where applicable)
# ---------------------------------------------------------------------------
def load_annotations(xlsx_path, recording_start_dt):
    """Parse Excel seizure annotation file into seconds relative to recording start.

    Reused from preprocessing_binary.py -- same logic, same column names.
    The annotation .xlsx has columns 'start_time' and 'end_time' in
    DD/MM/YYYY HH:MM:SS.fff format. We convert to seconds from the EDF
    recording start for direct comparison with segment grid positions.
    """
    df = pd.read_excel(xlsx_path)
    df['start_time'] = pd.to_datetime(df['start_time'], dayfirst=True)
    df['end_time'] = pd.to_datetime(df['end_time'], dayfirst=True)
    intervals = []
    for _, row in df.iterrows():
        start_dt = row['start_time'].to_pydatetime().replace(tzinfo=None)
        end_dt = row['end_time'].to_pydatetime().replace(tzinfo=None)
        start_sec = (start_dt - recording_start_dt).total_seconds()
        end_sec = (end_dt - recording_start_dt).total_seconds()
        intervals.append((start_sec, end_sec))
    return intervals


def find_edf_for_mouse(mouse_id):
    """Search all EDF directories for {mouse_id}.edf."""
    for edf_dir in EDF_DIRS:
        edf_path = edf_dir / f"{mouse_id}.edf"
        if edf_path.exists():
            return edf_path
    return None


def is_ictal(seg_start_sec, seg_end_sec, seizure_intervals):
    """True if the segment overlaps any seizure interval."""
    for sz_start, sz_end in seizure_intervals:
        if seg_start_sec < sz_end and seg_end_sec > sz_start:
            return True
    return False


def in_preictal_window(seg_start_sec, seg_end_sec, seizure_intervals,
                       lead_sec=PREICTAL_LEAD_SEC,
                       width_sec=PREICTAL_WIDTH_SEC):
    """True iff the segment is FULLY CONTAINED in [T - lead - width, T - lead]
    for at least one seizure onset T.

    Full-containment rule:
        seg_start >= T - lead - width   AND   seg_end <= T - lead

    Seizures whose onset T satisfies T < lead + width have no usable
    pre-ictal window (it would start before the recording) and are
    skipped here. Other annotated seizures in the same recording
    remain eligible.
    """
    for sz_start, _sz_end in seizure_intervals:
        window_start = sz_start - lead_sec - width_sec
        window_end = sz_start - lead_sec
        if window_start < 0:
            continue                                # seizure too early -- no usable window
        if seg_start_sec >= window_start and seg_end_sec <= window_end:
            return True
    return False


# ---------------------------------------------------------------------------
# Seizure segment verification plots
# ---------------------------------------------------------------------------
def plot_seizure_verification(splits, output_dir, n_samples=3, seed=42):
    """Randomly select and plot seizure segments from train, val, and test.

    For each partition that contains ictal segments, n_samples segments
    are randomly drawn, loaded from disk, and plotted as time-domain
    waveforms. This provides a visual sanity check that seizure segments
    contain plausible ictal EEG morphology rather than artefacts.
    Plots are saved to output_dir and NOT displayed (Agg backend).
    """
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    partitions = [
        ("train", splits.get("train", [])),
        ("val",   splits.get("val", [])),
        ("test",  splits.get("test", [])),
    ]

    for part_name, records in partitions:
        ictal_recs = [r for r in records if r["label"] == 1]

        if not ictal_recs:
            logger.info("Seizure verification: %s -- no ictal segments, skipping", part_name)
            continue

        k = min(n_samples, len(ictal_recs))
        sampled = rng.sample(ictal_recs, k)

        fig, axes = plt.subplots(k, 1, figsize=(14, 3 * k))
        if k == 1:
            axes = [axes]

        for idx, (ax, rec) in enumerate(zip(axes, sampled)):
            filepath = rec["filepath"]
            fname = Path(filepath).stem
            mouse_id = fname.split("_", 1)[0]

            try:
                segment = np.load(filepath)
                time_axis = np.arange(len(segment)) / FS

                ax.plot(time_axis, segment, color="#5A7DC8", linewidth=0.6)
                ax.set_title(
                    "%s | %s | shape=%s | min=%.2f | max=%.2f | std=%.2f" % (
                        part_name.upper(), fname, segment.shape,
                        segment.min(), segment.max(), segment.std()),
                    fontsize=9)
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Amplitude (z-scored)", fontsize=8)
                ax.tick_params(labelsize=7)

                ax.axvspan(0, time_axis[-1], alpha=0.08, color="#C85A5A",
                           label="Ictal segment")
                ax.legend(fontsize=7, loc="upper right")

            except Exception as e:
                ax.set_title("%s | %s | LOAD FAILED: %s" % (
                    part_name.upper(), fname, e), fontsize=9, color="red")
                logger.warning("Could not load %s: %s", filepath, e)

        plt.tight_layout()
        fig_path = output_dir / ("seizure_verification_%s.png" % part_name)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Seizure verification: %s -- %d segments plotted -> %s",
                    part_name, k, fig_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("create_T_120_splits.py")
    logger.info("Timestamp           : %s", datetime.now().isoformat())
    logger.info("Purpose             : Pre-ictal window [T-120, T-60] non-ictal selection")
    logger.info("Pre-ictal lead  sec : %.1f", PREICTAL_LEAD_SEC)
    logger.info("Pre-ictal width sec : %.1f", PREICTAL_WIDTH_SEC)
    logger.info("Amplitude threshold : %.0f", AMPLITUDE_THRESHOLD)
    logger.info("=" * 60)

    random.seed(SEED)
    np.random.seed(SEED)

    # -- Step 1: Load original data_splits.json --------------------------------
    if not SPLITS_PATH.exists():
        logger.error("data_splits.json not found at %s", SPLITS_PATH)
        raise FileNotFoundError(str(SPLITS_PATH))

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_records = splits["train"]
    val_records = splits["val"]
    logger.info("Loaded data_splits.json: %d train, %d val", len(train_records), len(val_records))

    # -- Step 1b: Visual verification of seizure segments ----------------------
    plot_seizure_verification(splits, OUTPUT_PATH.parent / "seizure_verification_plots",
                              n_samples=3, seed=SEED)

    # -- Step 2: Group train records by mouse ----------------------------------
    mouse_records = {}
    for rec in train_records:
        mouse_id = Path(rec["filepath"]).stem.split("_", 1)[0]
        mouse_records.setdefault(mouse_id, []).append(rec)

    logger.info("Unique mice in train: %d", len(mouse_records))

    # -- Step 3: Remove m254 ---------------------------------------------------
    if "m254" in mouse_records:
        n_removed = len(mouse_records.pop("m254"))
        logger.info("Removed m254: %d segments (ictal-free subject)", n_removed)

    # -- Step 4: Per-mouse pre-ictal classification ----------------------------
    # Keep ALL ictal. For non-ictal, keep only segments FULLY CONTAINED in
    # [T - 120, T - 60] for some seizure onset T in the same recording.
    # All other non-ictal (post-ictal, inter-ictal, immediate pre-ictal <60s)
    # are dropped.
    kept_ictal = []
    kept_preictal = []
    n_seizures_skipped_early = 0               # seizures with onset < lead+width (no usable window)
    n_mice_missing_meta = 0                    # mice with missing EDF or xlsx (defensive fallback)

    t0 = time.time()
    for mouse_id in sorted(mouse_records.keys()):
        records = mouse_records[mouse_id]
        ictal_recs = [r for r in records if r["label"] == 1]
        nonictal_recs = [r for r in records if r["label"] == 0]

        if not ictal_recs:
            logger.warning("Skipping %s: no ictal segments", mouse_id)
            continue

        edf_path = find_edf_for_mouse(mouse_id)
        xlsx_path = ANNOTATION_DIR / f"{mouse_id}_xlsx.xlsx"

        if edf_path is None or not xlsx_path.exists():
            # Pre-ictal identification requires annotations. Without them we
            # cannot determine which non-ictal segments fall in Window A, so
            # we DROP all non-ictal for this mouse (ictal still retained).
            # With the current cohort every mouse has an annotation file, so
            # this branch should never fire -- kept as a defensive guard.
            n_mice_missing_meta += 1
            logger.warning("%s: EDF or annotation not found. Dropping all non-ictal "
                           "(%d segments); ictal retained.", mouse_id, len(nonictal_recs))
            kept_ictal.extend(ictal_recs)
            continue

        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        n_samples = int(raw.n_times)
        rec_start = raw.info['meas_date'].replace(tzinfo=None)
        del raw

        seizure_intervals = load_annotations(xlsx_path, rec_start)

        # Count seizures whose onset is too early for a usable Window A
        min_onset = PREICTAL_LEAD_SEC + PREICTAL_WIDTH_SEC
        skipped_in_mouse = sum(1 for (s, _) in seizure_intervals if s < min_onset)
        n_seizures_skipped_early += skipped_in_mouse

        # Rebuild the segment grid and classify each non-ictal grid position
        segment_starts = np.arange(0, n_samples - WIN_LEN + 1, STEP, dtype=np.int64)

        # Map: 1-based non-ictal file index -> True if the corresponding grid
        # position is fully contained in some seizure's Window A
        index_in_preictal = {}
        nonictal_idx = 0
        for seg_start_idx in segment_starts:
            seg_start_sec = seg_start_idx / FS
            seg_end_sec = (seg_start_idx + WIN_LEN) / FS
            if is_ictal(seg_start_sec, seg_end_sec, seizure_intervals):
                continue
            nonictal_idx += 1
            index_in_preictal[nonictal_idx] = in_preictal_window(
                seg_start_sec, seg_end_sec, seizure_intervals,
                lead_sec=PREICTAL_LEAD_SEC, width_sec=PREICTAL_WIDTH_SEC)

        kept_preictal_mouse = 0
        for rec in nonictal_recs:
            fname = Path(rec["filepath"]).stem
            parts = fname.rsplit("_", 1)
            try:
                seg_index = int(parts[-1])
            except ValueError:
                # Can't parse index -- drop (cannot verify window membership)
                continue

            if index_in_preictal.get(seg_index, False):
                kept_preictal.append(rec)
                kept_preictal_mouse += 1
            # else: drop (post-ictal, inter-ictal, or immediate <60 s pre-ictal)

        kept_ictal.extend(ictal_recs)

        logger.info("  %s: %d ictal | %d preictal kept | %d nonictal dropped | "
                    "%d seizures | %d early-onset skipped | grid=%d",
                    mouse_id, len(ictal_recs), kept_preictal_mouse,
                    len(nonictal_recs) - kept_preictal_mouse,
                    len(seizure_intervals), skipped_in_mouse, len(segment_starts))

    elapsed_classify = time.time() - t0
    logger.info("Classification time: %.1f s", elapsed_classify)
    logger.info("Seizures skipped (onset < %.0f s into recording): %d",
                PREICTAL_LEAD_SEC + PREICTAL_WIDTH_SEC, n_seizures_skipped_early)
    if n_mice_missing_meta:
        logger.warning("Mice with missing EDF/annotation (non-ictal dropped): %d",
                       n_mice_missing_meta)

    # -- Step 5: Amplitude / NaN / Inf filter ----------------------------------
    # Same threshold (1000.0) used in filter_extreme_segments() in tcn_utils.py.
    # Segments with max|x| > 1000 indicate failed z-score normalisation from
    # interrupted preprocessing jobs. Also catches NaN/Inf values.
    all_kept = kept_ictal + kept_preictal
    logger.info("Pre-filter total: %d", len(all_kept))

    t0_filter = time.time()
    clean = []
    n_extreme = 0
    for i, rec in enumerate(all_kept):
        try:
            x = np.load(rec["filepath"])
            if not np.isfinite(x).all() or float(np.abs(x).max()) > AMPLITUDE_THRESHOLD:
                n_extreme += 1
                continue
            clean.append(rec)
        except OSError:
            n_extreme += 1
            logger.warning("Could not read %s -- excluded", rec["filepath"])
            continue
        if (i + 1) % 50000 == 0:
            logger.info("  filter_extreme: scanned %d/%d | removed: %d",
                        i + 1, len(all_kept), n_extreme)

    logger.info("Extreme segments removed: %d (%.2f%%)",
                n_extreme, 100 * n_extreme / max(len(all_kept), 1))
    logger.info("Filter time: %.1f s", time.time() - t0_filter)

    # -- Step 6: Build final train list ----------------------------------------
    final_train = clean
    n_ictal_final = sum(1 for r in final_train if r["label"] == 1)
    n_nonictal_final = sum(1 for r in final_train if r["label"] == 0)

    logger.info("=" * 60)
    logger.info("FINAL TRAINING CORPUS")
    logger.info("  Ictal                        : %d", n_ictal_final)
    logger.info("  Non-ictal (preictal window)  : %d", n_nonictal_final)
    logger.info("    Window [T-%.0f, T-%.0f] per seizure, full containment",
                PREICTAL_LEAD_SEC + PREICTAL_WIDTH_SEC, PREICTAL_LEAD_SEC)
    logger.info("    Extreme removed            : %d", n_extreme)
    logger.info("  Total                        : %d", len(final_train))
    if n_nonictal_final >= n_ictal_final:
        logger.info("  Ratio (ictal:non-ictal)      : 1:%.2f",
                    n_nonictal_final / max(n_ictal_final, 1))
    else:
        logger.info("  Ratio (non-ictal:ictal)      : 1:%.2f",
                    n_ictal_final / max(n_nonictal_final, 1))
    logger.info("=" * 60)

    # -- Step 7: Save output JSON ----------------------------------------------
    # The output JSON has the same schema as data_splits.json so all pipeline
    # scripts can load it without modification (same "train" and "val" keys).
    # Metadata records all parameters for reproducibility and audit.
    output = {
        "metadata": {
            "created_by": "create_T_120_splits.py",
            "timestamp": datetime.now().isoformat(),
            "source": str(SPLITS_PATH),
            "selection_rule": "preictal_window_T_minus_120_to_T_minus_60_full_containment",
            "preictal_lead_sec": PREICTAL_LEAD_SEC,
            "preictal_width_sec": PREICTAL_WIDTH_SEC,
            "amplitude_threshold": AMPLITUDE_THRESHOLD,
            "seed": SEED,
            "n_train_ictal": n_ictal_final,
            "n_train_nonictal": n_nonictal_final,
            "n_val": len(val_records),
            "excluded_subjects": ["m254"],
            "n_extreme_removed": n_extreme,
            "n_seizures_skipped_early_onset": n_seizures_skipped_early,
            "n_mice_missing_meta": n_mice_missing_meta,
        },
        "train": final_train,                         # list of {"filepath": ..., "label": 0/1}
        "val": val_records,                           # val partition unchanged from original
    }

    # Preserve test partition if it exists in original splits (for future use)
    if "test" in splits:
        output["test"] = splits["test"]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved: %s", OUTPUT_PATH)

    # -- Step 8: Summary -------------------------------------------------------
    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("  Train segments : %d", len(final_train))
    logger.info("  Val segments   : %d (unchanged)", len(val_records))
    logger.info("  Output         : %s", OUTPUT_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
