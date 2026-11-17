# User-supplied HeyGen quality reference — 2026-11-05

Reference: `/path/to/your/local/media/Timeline 1.mov` (the initially supplied extensionless path resolves to this MOV). This is third-party generated reference footage, never an Avatar Studio output and never a training dataset by implication.

FFmpeg metadata: 1920×1080, H.264, 60 fps, stereo 48 kHz AAC, roughly 14.9 seconds, DaVinci Resolve export. Export settings do not establish native model generation resolution or frame rate.

Sampled full frames at 1, 3, 5, 8, 12 seconds show an upper-body black-T-shirt presenter in a composed studio background. Visible target details: coherent moustache/lip border, stronger facial texture than our rejected output, consistent torso appearance, restrained pose variation. Static samples alone do not establish perceptual audio synchronization or all temporal artifacts.

Current failures to address:
- MuseTalk broad lower-face blend softens moustache, cheeks, and chin.
- The experimental tight mask preserves more source texture but introduces mouth seams/ghost edges; rejected as default.
- Current webcam and reference differ in composition, lighting and apparent source appearance. Fair comparisons require identifying the exact photo/video and voice-over used in HeyGen.

Reference stills/contact sheet are private local artifacts under `.local/heygen-reference/`. No HeyGen media is used as generated output or silently fed into training.

LatentSync status: previous queued run timed out after one hour. The Xet partial checkpoint has the expected file length but failed SHA-256; it must not be loaded. Download completion must be determined by integrity verification, never file length alone. Full pretrained LatentSync video inference is still unverified.

Inspection of `Portrait_BlackT.png` shows closely matching outfit/background to the reference. This suggests it may be the source image, but user confirmation is pending. A matched photo/audio/composition comparison should replace webcam-versus-reference comparisons.

User confirmed HeyGen received exactly `my_video.mp4` and `Portrait_BlackT.png`. Avatar Studio's matched-input test now uses the photo for appearance and the camera footage for motion. The exported HeyGen audio is extracted only as the driving-audio reference; HeyGen-generated visual frames are not passed off as Avatar Studio output. The exact internal HeyGen model/workflow remains unknown.

Matched local motion test completed: LivePortrait + MediaPipe adapter, Portrait_BlackT appearance, normalized first two seconds of my_video motion, 50 output frames at 25 fps, 1280×852 paste-back. Neural animation stage took approximately 1m45s. Artifact `.local/matched-reference/motion/Portrait_BlackT--driving.mp4` has no driving audio and is not a completed audio-driven avatar result. Sampled frames retain outfit/background, with restrained expression changes. Lip-sync comparison is a separate stage using a two-second reference-audio excerpt.

Matched two-stage inference completed: LivePortrait photo/motion transfer followed by MuseTalk v1.5, 50 generated frames at 25 fps with the two-second HeyGen reference-audio excerpt. Full decode/frame-count validation and independent decoded-audio equality passed. Candidate: `.local/matched-reference/lipsync/export.mov`. The visual source is locally animated photo footage, not raw camera frames; provenance explicitly records this. Sampled output still exhibits lower-face softness, so HeyGen-level quality is not achieved. This is a benchmark candidate, not an approved production result.
