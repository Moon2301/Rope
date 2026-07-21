<img width="2343" height="1375" alt="Screenshot 2026-07-11 132434" src="https://github.com/user-attachments/assets/abb3cfe1-9ec6-4ce5-b8fe-8c9de7a45294" />

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

### Disclaimer: ###
Rope is a personal project that I'm making available to the community as a thank you for all of the contributors ahead of me.
I've copied the disclaimer from [Swap-Mukham](https://github.com/harisreedhar/Swap-Mukham) here since it is well-written and applies 100% to this repo.
 
I would like to emphasize that our swapping software is intended for responsible and ethical use only. I must stress that users are solely responsible for their actions when using our software.

Intended Usage: This software is designed to assist users in creating realistic and entertaining content, such as movies, visual effects, virtual reality experiences, and other creative applications. I encourage users to explore these possibilities within the boundaries of legality, ethical considerations, and respect for others' privacy.

Ethical Guidelines: Users are expected to adhere to a set of ethical guidelines when using our software. These guidelines include, but are not limited to:

Not creating or sharing content that could harm, defame, or harass individuals. Obtaining proper consent and permissions from individuals featured in the content before using their likeness. Avoiding the use of this technology for deceptive purposes, including misinformation or malicious intent. Respecting and abiding by applicable laws, regulations, and copyright restrictions.

Privacy and Consent: Users are responsible for ensuring that they have the necessary permissions and consents from individuals whose likeness they intend to use in their creations. We strongly discourage the creation of content without explicit consent, particularly if it involves non-consensual or private content. It is essential to respect the privacy and dignity of all individuals involved.

Legal Considerations: Users must understand and comply with all relevant local, regional, and international laws pertaining to this technology. This includes laws related to privacy, defamation, intellectual property rights, and other relevant legislation. Users should consult legal professionals if they have any doubts regarding the legal implications of their creations.

Liability and Responsibility: We, as the creators and providers of the deep fake software, cannot be held responsible for the actions or consequences resulting from the usage of our software. Users assume full liability and responsibility for any misuse, unintended effects, or abusive behavior associated with the content they create.

By using this software, users acknowledge that they have read, understood, and agreed to abide by the above guidelines and disclaimers. We strongly encourage users to approach this technology with caution, integrity, and respect for the well-being and rights of others.

Remember, technology should be used to empower and inspire, not to harm or deceive. Let's strive for ethical and responsible use of deep fake technology for the betterment of society.



  
