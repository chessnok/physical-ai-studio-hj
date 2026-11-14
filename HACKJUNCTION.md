# HackJunction2026 fork of Physical AI Studio

Private copy of [open-edge-platform/physical-ai-studio](https://github.com/open-edge-platform/physical-ai-studio),
branch `junction2026` @ `17714488` (2026-11-13), with our changes on top. Team repo with the project
docs (ADRs, troubleshooting, daily log): `chessnok/HackJunction2026` (ADR 0004, T-011 – T-013).

## Branches

| Branch | Contents |
| --- | --- |
| `junction2026` | Upstream, unchanged (full history) |
| `hj/realsense-rgbd` | 13 commits: RealSense color + aligned depth recording, Studio-run camera publishers, review fixes |
| `hj/state-snapshots` | `hj/realsense-rgbd` + 3 commits: Space-hotkey state snapshots, + this notes commit. **Use this one.** |

## Run it (native, as in JUNCTION2026.md)

```bash
git clone -b hj/state-snapshots git@github.com:chessnok/physical-ai-studio-hj.git
cd physical-ai-studio-hj/application/backend && cp -n .env.example .env
uv run --extra xpu physicalai-studio serve            # :7860
cd ../ui && npm install && npm run start               # :3000 (Node >= 24.2, npm >= 11.14)
```

On our laptop the NPU also needs
`ZE_ENABLE_ALT_DRIVERS=/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1,/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu/libze_intel_npu.so.1`
(list both drivers; only the NPU one breaks `torch.xpu`). The team repo's `scripts/studio_backend.sh` does that.
No new dependencies and no DB migration compared to `junction2026`.

## What changed

### RealSense RGB-D recording (`0001–0006` + fixes `0007–0013`)
- Per camera `output_type`: `color` (unchanged, backward compatible), `depth`, `both`; depth range
  filter (`depth_range_min/max`, outside → 0 = no data).
- Features: `observation.images.<cam>` (color), `observation.images.<cam>_depth` (LeRobot 0.6 native
  depth feature, [H, W, 1], millimetres, `info.is_depth_map`).
- Storage: 12-bit log quantization over 0–10 m (0 stays 0; step about 1.3 mm at 0.3 m, about 2.1 mm at 3 m), lossless HEVC
  `gray12le` (libx265, ultrafast). Round trip measured at 1.0 mm maximum error.
- **Depth is recorded at half the color resolution by default** (`DEPTH_RECORD_DOWNSCALE=2`, 640×360
  for 1280×720 color; depth pixel (u, v) = color pixel (2u, 2v), real samples, not interpolated).
  Full-resolution lossless depth for two cameras could not keep up and LeRobot silently drops frames.
  `DEPTH_RECORD_DOWNSCALE=1` gives full resolution, but expect dropped and discarded episodes.
- Episodes are **discarded with a UI message** when an encoder drops a frame or rows fall more than
  0.2 s behind the dataset clock; after saving, video frame counts are checked against row counts.
- Depth aligned to color (`rs.align`), one frameset → one timestamp; store reads use a sequence check
  (no torn frames; relies on x86 memory ordering).
- Camera publishers are started and supervised by Studio (one per device): settings changes replace
  the publisher (newest request wins), dead/phantom publishers are replaced without a restart, orphans
  from a dead backend are stopped via an on-disk registry (pid + start time), backend has a 10 s
  graceful-shutdown timeout. The same device twice in one environment is refused (422).
- Training excludes depth features (`exclude_depth`); single-image policies read a color camera.
- UI: output type and depth range in the RealSense form, colorized depth previews/thumbnails,
  depth keys hidden from camera mappings.

### State snapshots (`snapshots/0001–0003`)
- Hotkey action "Capture state snapshot" (default Space, rebindable in Settings), in environment,
  recording and inference views; ignored while typing.
- `POST /api/projects/{id}/state_snapshots?session_id=…` (409 `no_live_session` without a session),
  `GET` lists them. Read-only copy of the session store: no device opened, no robot command.
- Saved to `<storage>/state_snapshots/<project>/<YYYYmmdd-HHMMSS-mmm>/`: `state.json` (times,
  joints, ticks from calibration, robots + calibration copy, camera settings, RealSense intrinsics,
  depth scale, version), `<cam>.color.png`, `<cam>.depth.png` (uint16 mm), `<cam>.depth_preview.png`.

## Review history

1. **Implementation** (agent 1): `0001–0006`. Live check: both D415 at 1280×720@30 color + aligned
   depth at 30.0 / 29.7 fps; 6 s recording 180/180 frames; resolution change and dead-publisher
   recovery without restart.
2. **Independent review** (agent 2, code + own experiments, no hardware): verdict **"deploy after
   specific fixes"**. Major: full-res lossless depth uses about 9 of 16 cores for two cameras; when the
   encoder falls behind, LeRobot drops depth frames silently → shorter, misaligned depth video and a
   time-compressed episode (reproduced with synthetic load). Minor: no warning for a depth camera
   recording into a dataset without depth features; same device twice loops forever; pattern-based
   orphan sweep could kill other programs' publishers; single-image policies could read depth; ~1 %
   torn frames; untracked-publisher race; stale settings request could win; edge cases. Verified OK:
   depth round trip, training exclusion, color-only compatibility, publisher recovery, SIGTERM fix.
3. **Fixes** (agent 1): `0007–0013`, all must- and should-fix items addressed. Full target load
   (1080p + 2×720p color + 2 depth, 30 fps) with real D415 depth at half resolution: 0 drops,
   900/900 frames per video, about 8.4 cores, about 103 MB/min per depth camera. Rejected alternatives:
   x264 10-bit lossless (inexact readback), FFV1 (not in mp4), libaom (slow, inexact), x265 lossy
   (slower). Side finding: libx265 mp4 muxing of 1–2 frame streams fails randomly, so episodes need
   at least 3 frames and the encoder check uses 4.
4. **Re-review of the fixes and snapshots: started, then stopped by the team before it finished.**
   The fixes (`0007–0013`) and the snapshot feature are **not independently re-verified**.

Tests reported by the implementation agent (not re-run by a reviewer for the fixes): backend
2233 passed (`hj/realsense-rgbd`) / 2243 passed (`hj/state-snapshots`), UI 589 / 595 passed, ruff and
pyrefly clean, mypy only the pre-existing `src/schemas/job.py:17`.

## Not tested on hardware yet

- Half-resolution depth recording through the UI, and discard-on-drop/lag in a real session.
- The snapshot hotkey with real cameras and robots.
- The SIGTERM fix against a live stuck backend (unit-tested only).
- Docker/LGPL builds and remote trainers (an older remote trainer library fails on `exclude_depth`).

## Suggested acceptance test on the robot laptop

1. Checkout `hj/state-snapshots`, restart the backend. If a camera reports "served by publisher process
   N, which this Studio backend did not start", `kill N` (old-backend publishers are not in the registry).
2. Both D415 with driver RealSense, output "Color + depth", 1280×720 @ 30, depth range e.g. 0.3–1.5 m;
   both on USB 3.
3. Record into a **new** dataset at 30 fps for 2–3 minutes with all three cameras; check the backend
   log for "Encoder queue full", "dropped", "falls behind", discarded episodes; open an episode and
   check the depth preview.
4. Press Space in the environment view; check `~/.local/share/physicalai/state_snapshots/…`.
5. Change a camera's resolution in the form while previewing; the preview must recover.
