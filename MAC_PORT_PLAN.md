# Mac hardening and port plan

Branch: `mac-hardening` (this repo) and `mac-hardening` in the `external/unsloth-zoo` submodule
(fork: FiditeNemini/unsloth-zoo). This is a **hard fork**: non-Mac code and telemetry are deleted,
not gated. Upstream fixes are cherry-picked by hand.

## Goals

1. **No telemetry, no phone-home.** Nothing leaves the machine that the user did not ask for.
2. **Offline by default.** Every outbound network path (HF Hub, ModelScope, GitHub, PyPI,
   external LLM providers, wandb, tunnels, research tools) is off until the user turns it on in
   Settings. The default install works fully air-gapped with local models.
3. **Self-contained.** No runtime `pip install`, no downloading binaries or wheels at runtime.
   Dependencies are resolved at install time and locked.
4. **Mac-only.** Apple Silicon, macOS. Backends: MLX / mlx-lm / mlx-vlm for training and
   inference, llama.cpp (Metal) for GGUF inference, stable-diffusion.cpp / MLX for image gen.
   CUDA, ROCm, XPU, NPU, Triton, vLLM, bitsandbytes, Windows and Linux are removed.

## Codebase at a glance

| Tree | Non-test Python LOC | Tests |
|---|---|---|
| `unsloth/` | ~77k (9k of it `kernels/`, Triton) | 812 |
| `studio/backend/` | ~429k | 1064 |
| `external/unsloth-zoo/unsloth_zoo/` | ~154k (53k of it `mlx/`) | 360 |

Files mentioning each platform (non-test): cuda 177, windows/msvc 203, triton 105,
rocm/hip 103, bitsandbytes 104, xpu 83, vllm 53, mlx 92. (`npu` over-matches; ignore.)

What already exists for Mac, and is the foundation:
- `unsloth/device_type.py` has an MLX runtime (`DEVICE_TYPE == "mlx"`, torch never imported).
- `unsloth_zoo/mlx/` — loader, trainer, inference, generate, CCE loss, quantized optimizers.
- `studio/backend/core/inference/mlx_inference.py`, `llama_cpp.py`, `sd_cpp_*.py`.
- `studio/backend/utils/hardware/hardware.py` — `is_apple_silicon`, CHAT_ONLY gating.
- `studio/src-tauri/` — Tauri desktop app with a macOS config, entitlements and DMG build.
- zoo pins: `mlx==0.32.2`, `mlx-lm==0.31.3`, `mlx-vlm>=0.4.4,<=0.7.1` (darwin/arm64).

---

## Phase 1 — Network policy module (foundation) — done

Everything later depends on a single place that answers "may this process talk to X?".

**Status.** Built as planned, with these specifics:
- `unsloth_network_policy` is a stdlib-only **top-level** module shipped in the zoo
  distribution (`py-modules`), so Studio's light parent imports it without running
  `unsloth_zoo/__init__.py`, which activates it first thing.
- Studio: `utils/network_policy_settings.py` stores it under `app_settings.network_policy` in
  the owner's `studio.db`; `main.py` applies it before anything imports HF libraries, and the
  **stored value beats an inherited `UNSLOTH_NETWORK_POLICY`**. Every spawned worker goes
  through `utils/native_path_leases.run_without_native_path_secret`, which activates it and
  fails closed if the module is missing.
- API: `GET /api/settings/network-policy` (everyone), `PUT` (owner, UI session only — an
  API key gets 403). UI: Settings → General → Network access (master switch, a switch per
  service, and "Local network").
- `--local` installs take zoo from `external/unsloth-zoo`, editable.
- Not yet: the `sandbox-exec` end-to-end smoke test, and the per-service `require()` calls
  (Phase 3). Subprocesses (git, curl, llama-server) are outside the socket guard.

**Design**
- New `unsloth_zoo/network_policy.py` (lowest layer: both `unsloth` and Studio depend on zoo).
  - Services enum: `hf_hub`, `modelscope`, `github`, `pypi`, `llm_providers`, `wandb`,
    `tunnel`, `web_research` (YouTube, arXiv, web search tools), `remote_media` (http(s)
    image/video URLs in datasets), `mcp_remote`.
  - `allowed(service) -> bool`, `require(service)` (raises `NetworkDisabledError` with a
    message naming the Settings toggle).
  - Source of truth: env var `UNSLOTH_NETWORK_POLICY` (JSON) so Studio worker subprocesses
    inherit it. Absent/invalid ⇒ **everything off**.
- **Enforcement at startup** when a service is off:
  - `hf_hub` off ⇒ force `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
    `HF_DATASETS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, `DO_NOT_TRACK=1`.
  - Master off ⇒ install a socket guard (`socket.socket.connect`, `socket.getaddrinfo`,
    `socket.create_connection`) that refuses non-loopback destinations. This is the backstop
    for third-party libraries we don't control. Loopback stays open (Studio ↔ workers,
    llama-server, sd-server).
- **Studio setting**: stored in `app_settings` (`studio/backend/storage/studio_db.py`), exposed
  via `studio/backend/routes/settings.py`, one "Network access" master toggle + per-service
  toggles in the frontend Settings page. Changes restart workers so the env propagates.
- **CLI**: `unsloth` library honours the env var; `unsloth_cli` gets `--allow-network=hf,...`.

**Tests**: pytest fixture that installs the socket guard for the whole suite, plus a test per
service that its call site raises `NetworkDisabledError` when off. An end-to-end smoke test runs
Studio under `sandbox-exec` with a deny-network profile (loopback allowed) and exercises load,
chat, train, export on a local model.

## Phase 2 — Remove telemetry and phone-home — done

**Status.** Removed: the HF usage-statistics ping; the GitHub `mapper.py` refresh; the PyPI
update check and GitHub release-notes fetch (routes keep their shape, always "no update");
every GitHub release lookup behind llama.cpp/whisper.cpp freshness and the changelog (the two
fetchers in `utils/prebuilt/freshness_flow.py` answer "unknown"; caches from the fetching era
are ignored); the startup freshness check; the latest-transformers PyPI/GitHub probe; the
public-IP lookups (`ifconfig.me`, GCE metadata) and the **check-host.net** reachability probe
(it sent your public IP:port to a third party; missed by the first inventory); the Xet
reachability probe; the ROCm wheel-index probe; the `chat.cpp` fetch (now read from the
installed llama.cpp source tree); Trackio's remote logos; and the CLI's fallback that
downloaded `unsloth.ai/install.sh` and piped it into bash (it would have replaced this fork
with upstream).

Deferred, by design:
- In-app llama.cpp/whisper.cpp updater and backend switch (`utils/llama_cpp_update.py`,
  `routes/llama.py`, `routes/whisper.py`, ~14 frontend files): now inert (it can never find a
  release) but its apply endpoint would install a prebuilt over the Metal source build if
  called directly. **First item of Phase 4.**
- Tauri auto-updater: Phase 5 (Rust shell + desktop update UI). The installed
  `/Applications/Unsloth.app` is upstream's build and still polls GitHub until rebuilt.
- CLI processes do not activate the network policy yet, and `unsloth start <model>` /
  `unsloth start hermes` reach HF/GitHub: Phase 3.

Original list:

| What | Where |
|---|---|
| Usage statistics ping (HF `unslothai/{platform,vram,gpu-count}` download) | `unsloth/models/_utils.py` `_get_statistics`/`get_statistics`; call sites `models/llama.py:~2438`, `models/vision.py:~1391` |
| Remote model-mapper refresh (GitHub raw `mapper.py`) | `unsloth/models/loader_utils.py:_get_new_mapper` and call site ~1030 |
| Public IP lookup (`ifconfig.me`, GCE metadata) | `studio/backend/run.py:_resolve_external_ip` |
| PyPI update check, release-notes fetch (GitHub API) | `studio/backend/utils/update_status.py`, `utils/release_notes.py`, routes in `main.py` |
| llama.cpp / whisper.cpp freshness + changelog (GitHub API, 24h) | `utils/prebuilt/freshness_flow.py`, `utils/llama_cpp_freshness.py`, `utils/llama_cpp_changelog.py`, startup thread `main.py:~484–532` |
| Latest-transformers probe (PyPI + GitHub raw) | `studio/backend/utils/transformers_latest.py` |
| Xet reachability probe | `unsloth_zoo/hf_xet_health.py`, `studio/backend/utils/hf_xet_fallback.py` |
| ROCm wheel index probe | `unsloth_zoo/device_type.py:_pytorch_rocm_index_exists` (goes with Phase 5 anyway) |
| llama.cpp `chat.cpp` quirk check (GitHub raw) | `unsloth_zoo/llama_cpp.py:_check_llama_cpp_appended_system_message` — replace with a static answer |
| Tauri auto-updater (polls `github.com/unslothai/unsloth/releases/latest`) | `studio/src-tauri/tauri.conf.json` `plugins.updater` |
| Remote-code "trusted org" lookups | `studio/backend/utils/security/trusted_org.py`, `remote_code_approvals.py` — review; keep local-only logic |

Also remove the now-dead env switches (`UNSLOTH_DISABLE_STATISTICS`,
`UNSLOTH_DISABLE_UPDATE_CHECK`, `UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK`, `UNSLOTH_DISABLE_XET`,
`UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS`, `UNSLOTH_OFFLINE_PROBE`) and their tests.

## Phase 3 — Gate user-initiated egress behind the policy

These stay as features but call `require(service)` first, and the UI hides/greys them when off.

- **HF Hub / ModelScope** (`hf_hub` / `modelscope`): model load paths in
  `unsloth/models/loader.py`, `loader_utils.py`, `vision.py`, `sentence_transformer.py`,
  `unsloth/utils/hf_hub.py`, `unsloth/registry/*`; push/export in `unsloth/save.py`,
  `unsloth_zoo/saving_utils.py`, `unsloth_zoo/mlx/utils.py`; Studio `hub/` (workers, services,
  routes), `routes/models.py`, `utils/models/model_config.py`, `core/export/export.py`,
  `utils/hf_token_validation.py`, `utils/datasets/*`. The model picker falls back to local
  inventory (`hub/services/models/local_inventory.py`, `cache_inventory.py`) and scan folders.
- **llama-server self-download flags** (`-hf`, `-mu`, `--hf-repo` in
  `core/inference/llama_server_args.py`): reject unless `hf_hub` is on.
- **Remote dataset media** (`remote_media`): `unsloth_zoo/vision_utils.py`
  `_stream_guarded_media` → `require("remote_media")`. Local paths, PIL, bytes, base64 unaffected.
- **External LLM providers** (`llm_providers`): `core/inference/external_provider.py`,
  `openai_codex_client.py`, `openai_codex_auth.py`, `routes/providers.py`, data-recipe LLM calls,
  `utils/datasets/llm_assist.py`. Consider deleting Codex OAuth entirely.
- **Research / tools** (`web_research`): `core/inference/tools.py` (78 call sites — web fetch,
  search), `core/youtube_transcript.py`, `core/research_runs.py`, the GitHub-repo-seed plugin.
- **MCP** (`mcp_remote`): remote MCP servers in `mcp_server.py` / `routes/mcp_servers.py`;
  local stdio servers remain allowed.
- **wandb** (`wandb`): `core/training/trainer.py` `report_to`. TensorBoard stays (local).
- **Cloudflare tunnel** (`tunnel`): `studio/backend/cloudflare_tunnel.py`, `run.py`. Probably
  delete; LAN access (`lan_access.py`) stays but defaults to loopback bind.
- **S3 datasets**: `core/training/s3_dataset.py` — gate or delete (boto3 is optional).
- **Remote-code third-party sources** (`utils/third_party_source.py`: Spark-TTS, OuteTTS GitHub
  archives): delete, or require pre-installed.

## Phase 4 — No automatic installs

Updates stay possible, but only when the user runs them (`unsloth studio update` /
`setup.sh`), from pinned sources, behind the network policy. What goes is Studio installing
or downloading things by itself while it runs: that leaks, and executes unreviewed code.
Replace each of these with "installed at setup time, or the feature reports it is unavailable":

- `studio/backend/utils/mlx_repair.py` (MLX autorepair, default-on) — delete; the installer
  guarantees the MLX stack.
- `studio/backend/utils/transformers_version.py` — the `.venv_t5_*` sidecar venvs are created
  on demand from PyPI/GitHub. Decide: pre-provision at install, or drop the sidecars and pin one
  transformers version that the MLX stack supports.
- `studio/backend/utils/wheel_utils.py` (flash-attn, PyTorch index, Unsloth prebuilt wheels),
  `utils/diffusers_repair.py`, `core/training/worker.py` causal-conv1d self-heal,
  `core/inference/diffusion_families.py`, `unsloth/import_fixes.py`, `unsloth/_auto_install.py`,
  `unsloth_zoo/temporary_patches/fla_vendor.py`, llmcompressor auto-install — delete.
- llama.cpp: **done for macOS** — see "llama.cpp source" below. Still to remove: the prebuilt
  path in `studio/install_llama_prebuilt.py` and the in-app updater/freshness checks
  (`utils/llama_cpp_update.py`, `utils/llama_cpp_freshness.py`), which only act on prebuilts.
- whisper.cpp: **done for macOS** — `setup.sh` builds it from source (Metal) via
  `scripts/build_whisper_cpp.sh`; see "whisper.cpp source" below.
- sd.cpp / STT sidecar downloaders (`core/inference/stt_download_worker.py`, `sd_cpp_*`) —
  same treatment.

### llama.cpp source

macOS builds llama.cpp from source with Metal, never from a prebuilt. The source is
**FiditeNemini/llama.cpp**, at a pinned *Unsloth mix* tag:

- Unsloth ships llama.cpp as upstream `bNNNN` + PRs pinned in `scripts/unsloth/pr-set.json`
  (new architectures, the IQ1_XS/XXS/XXXS quants, perf fixes), merged in CI and released only
  as binaries. The merged source is never pushed anywhere.
- `scripts/unsloth/make_mix_tag.sh` (on the fork's `unsloth-tools` branch, which tracks
  `unslothai/llama.cpp` master) runs the same merge locally and tags it with the same name
  Unsloth's release uses (`bNNNN-mix-<pr-set hash>`), plus a commit pinning
  `BUILD_NUMBER` to the base so `llama-server --version` reports `build NNNN`.
- `studio/setup.sh` defaults (`_DEFAULT_LLAMA_MAC_SOURCE/_TAG/_COMMIT`): clones the tag,
  **refuses to build unless HEAD is the pinned commit**, builds `llama-server`,
  `llama-quantize` and the DiffusionGemma visual server with Metal, and skips the rebuild
  when the installed binary already reports the pinned commit.
  `UNSLOTH_LLAMA_SOURCE` / `UNSLOTH_LLAMA_TAG` override (and drop the pin check).

Current pin: `b11160-mix-a6922cc` @ `457b94332494c08fc4073eea1606fb3f39bc1a8b`.

To move to a newer mix (in the llama.cpp clone):

```bash
git switch unsloth-tools && git fetch unsloth && git merge unsloth/master   # new pr-set.json
scripts/unsloth/make_mix_tag.sh            # or a specific base: make_mix_tag.sh b11200
git push origin unsloth-tools <tag>
```

then set `_DEFAULT_LLAMA_MAC_TAG` / `_DEFAULT_LLAMA_MAC_COMMIT` in `studio/setup.sh` to the
printed tag and SHA. Merge commits carry timestamps, so re-running the script yields a
different SHA: pin the one that was pushed. Then rebuild the whisper.cpp tag below on the new
llama tag, so the two keep sharing one ggml.

### whisper.cpp source

Unsloth's whisper.cpp has no source patches: its "slim" prebuilds are upstream `vX.Y.Z` with
`ggml/` replaced by the paired llama.cpp mix's ggml, linked at runtime against the llama
prebuilt's ggml dylibs. The source equivalent lives in **FiditeNemini/whisper.cpp**:

- `scripts/unsloth/make_ggml_tag.sh <whisper tag> <llama tag>` (fork's `unsloth-tools`
  branch, tracking `unslothai/whisper.cpp` master) commits that ggml swap onto the upstream
  tag and tags it `<whisper tag>-ggml-<llama tag>`.
- `scripts/build_whisper_cpp.sh` defaults to that tag, refuses to build unless HEAD is the
  pinned commit, builds a static `whisper-server` (its own copy of the shared ggml source, so
  the two installs cannot break each other), and skips the build when the stamped commit of
  the installed binary matches. `WHISPER_CPP_SOURCE` / `WHISPER_CPP_TAG` override.

Current pin: `v1.9.4-ggml-b11160-mix-a6922cc` @ `8358c3d153d022028ef0a137a2167dc18032daf6`.

To move (in the whisper.cpp clone, after the llama.cpp tag is pushed):

```bash
scripts/unsloth/make_ggml_tag.sh v1.9.4 <new llama tag>
git push origin unsloth-tools <tag>
```

then set `_DEFAULT_WHISPER_CPP_TAG` / `_DEFAULT_WHISPER_CPP_COMMIT` in
`scripts/build_whisper_cpp.sh`.

## Phase 5 — Frontend and desktop shell

- **CSP** (`studio/backend/main.py:~1070–1125` and `src-tauri/tauri.conf.json`):
  - `connect-src 'self'` only; add HF origins dynamically only when `hf_hub` is on.
  - `img-src`/`media-src` currently allow **any `https:`** — rendered chat markdown can load
    remote images (tracking pixels, and a prompt-injected model output could exfiltrate data in
    an image URL). Tighten to `'self' data: blob:`.
  - Drop the Colab branch.
- **Direct browser → HF calls** (bypass the backend, so the backend policy can't see them):
  `features/hub/lib/network.ts`, `hub-feed-store.ts`, `hf-readme.ts`, `dataset-size.ts`,
  `lib/hf-endpoint.ts`, model-picker `pickers.tsx`, `example-dataset-cards.tsx`,
  `export-run-panel.tsx`, `dataset-download-section.tsx`, `publish-execution-dialog.tsx`,
  `mcp-composer-button.tsx`. Route them through the backend (so one policy applies) or hide
  behind the `hf_hub` flag read from `/api/settings`.
- External links (`unsloth.ai`, docs, provider consoles): harmless when clicked, but remove
  from default UI surfaces and never auto-load. Fonts are already local (`font-src 'self' data:`).
- Tauri: remove updater; tighten capabilities; keep macOS config and DMG build; delete
  `tauri.linux.conf.json`, `tauri.windows.conf.json`, `src-tauri/linux`, `src-tauri/windows`.

## Phase 6 — Mac-only port

Largest phase; do it after 1–5 so the delete-heavy diffs don't bury the security work.
Order to keep the tree importable at every commit:

1. **Pin the runtime**: `DEVICE_TYPE` is always `mlx`; fail fast on non-darwin/arm64 at import.
2. **Studio backend**: remove CUDA/ROCm/XPU branches in `utils/hardware/`, training worker,
   inference orchestrator, diffusion stack (torch/diffusers paths → MLX / sd.cpp), vLLM,
   FSDP2, NVLink/P2P, `_msvc_env.py`, `_platform_compat.py` Windows code, Colab (`colab.py`).
3. **unsloth library**: delete `unsloth/kernels/` (Triton), `_gpu_init.py`,
   bitsandbytes/fp8/QGalore, vLLM/RL paths that need CUDA, `_compressed_quantize.py`; model
   classes that only exist to patch torch/CUDA. Decide whether the torch-on-MPS path is kept at
   all or MLX is the only training path (recommended: MLX only).
4. **unsloth-zoo**: keep `mlx/`, `llama_cpp.py`, `vision_utils.py` (data side), dataset/
   tokenizer/chat-template utils, saving (MLX + GGUF). Delete `vllm_*`, `temporary_patches/`
   for CUDA libs, `compiler*.py`, `flex_attention/`, `fused_losses/`, `gated_delta_vjp.py`,
   `tiled_mlp.py`, `fp16_emulation.py`, `device_map_planner.py`, `rl_environments.py` as
   they prove dead.
5. **Install / packaging**: keep `install.sh` + `studio/setup.sh` (macOS parts);
   delete `install.ps1`, `studio/setup.ps1`, `setup.bat`, `scripts/*rocm*`, `docker/` (all
   Linux/NVIDIA/ROCm), Windows/Linux release workflows. Produce a locked, offline-installable
   env (`uv lock` + wheelhouse) so installation itself is the only network step.
6. **CI**: of 58 workflows, keep lint + macOS test jobs; add the offline sandbox smoke test.

## Open decisions

- **Torch at all?** MLX-only removes torch from the training path, but some Studio features
  (audio/TTS, some diffusion families, embeddings for RAG) may still need torch-on-MPS.
  Inventory in Phase 6 step 2 before deciding.
- **transformers sidecar venvs** (Phase 4): pre-provision vs single pin.
- **Model catalogue when offline**: ship a local curated list, or local-inventory only.
- **Licensing**: `studio/` is AGPL-3.0; the rest is Apache-2.0 (`LICENSE`) with LGPL-3.0 zoo.
  Distributing the port obliges source availability for Studio changes — fine for a public fork.

## Commit / PR strategy

One PR per phase (per repo where zoo is touched), in order 1 → 6. Phases 1–3 are small and
reviewable; Phase 6 is split by subsystem. Each PR must pass the offline sandbox smoke test once
Phase 1 lands.
