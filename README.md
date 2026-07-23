Rope implements the insightface inswapper_128 model with a helpful GUI.
### Updates for Rope-Bronze: ###
* New, more responsive UI
* TRT Engine for better performance
* Batched inswapper for better 256 and 512 mode performance
* Settings tab for managing folders, models threading, benchmarking, ...
* New Likeness / Fidelity settings
* Color Matching (LAB) for accureate color matching
* XSeg masker
* Easier Embedding management. Drag and drop embeddings to reorder them.
* New Capture mode. Move and resize a window on your desktop to swap whatever is in it.
* Five-point temporal stabilization for smoother video alignment and safer identity tracking.

### Single-character Auto Job

1. Load a video, seek to the character, click **Find Faces**, and assign one source face or embedding.
2. Click **Start Auto Job**. Preflight checks the input, models, FFmpeg, output folder, free space, and face assignment before any GPU work starts.
3. The job scans or resumes its cache, then always stops at **AWAITING_REVIEW**. Review the start/middle/end thumbnails, warnings, frame bounds, split/merge edits, and selected swap duration.
4. Click **Confirm & Render**. There is no code path from scan to render without this explicit confirmation.
5. Rope renders minute-long video-only parts, joins them, muxes the original audio once, and runs output QC. Use **Open Output** after the job reaches **COMPLETED**.

Auto Jobs persist under the application-data directory. Scan progress is
checkpointed inside each five-minute chunk every 100 model calls or ten
seconds. Render checkpoints are finalized every 60 seconds, so a cancel,
crash, or application restart only repeats the unfinished scan block or render
part. Open **Resume Auto Job** to continue.

**Temporal Stabilization** is enabled by default in the Parameters tab for
video preview and recording. RetinaFace/SCRFD and ArcFace still use their raw
five landmarks for detection and recognition; Rope filters only the resulting
alignment transform before swapping. Matching is one-to-one, scene cuts and
seeks reset the tracker, and a confident track may coast for at most one missed
detector frame. Auto Job render checkpoints also preserve tracker state across
60-second part boundaries.

Long-video scanning keeps the sensitive 0.5-second coarse pass and refines
every three frames within one second of hits and near-hits. Cache keys include
the video fingerprint, target embedding, detector settings, threshold, and
algorithm version. Borderline, sub-0.7-second, and single-hit ranges are marked
for extra review.

Auto rendering writes RGB frames directly to FFmpeg instead of wrapping every
frame as a BMP. A real one-frame probe selects `h264_nvenc` when it works and
falls back to `libx264`; a live NVENC failure retries the current part with
x264. Frames outside approved ranges are passed through without face detection
or swapping. QC verifies readable output, resolution, FPS, frame count, audio,
sampled black frames, and sampled source-identity similarity. Temporary parts
are removed only after QC succeeds.

### Install from scratch:
```cmd
py -3.12 -m venv venv
```
```cmd
venv\Scripts\activate
```
```cmd
pip install -r requirements.lock.txt
```
Also, copy models from the Rope-Bronze Models Release to somewhere on your drive. In settings, select the folder they were copied to (you have to unzip them).



  
