# BeamNG.Earth — agent guide

A 1:1 scale, streamed Earth for BeamNG.drive: a companion app (launcher + local world server) plus a game mod that
streams real elevation, imagery, roads and buildings around the player. Version is in `VERSION` (`1.8.2-beta`).
Repo: https://github.com/TecnaGamer/BeamNG.Earth. License MIT (`LICENSE`).

## Layout

| Path | What it is |
|---|---|
| `launcher.py` | One-start app: copies the mod, starts the server, starts the game (this is the shipped AppImage/.exe) |
| `start.py`, `install.py`, `apppaths.py`, `bmplaunch.py` | Dev server start, installer, app folder choices, BeamMP (multiplayer) launching |
| `run_server.{bat,sh}`, `run_viewer.{bat,sh}` | Player entry points |
| `server/` | `earthd.py` — the world server (Python glue, protocol 4, TCP 127.0.0.1:47613). ~200 modules; `server/README.md` documents every flag and the protocol |
| `server/fast.py` | Loader for the Rust kernels; `fast.require(MIN_VERSION, ...)` with **no Python fallback** |
| `server/earthfast.abi3.so` / `.pyd` | Prebuilt Rust module checked into the repo (Linux x86-64 + Windows) |
| `native/earthfast/` | The Rust geometry/image kernels (pyo3, abi3-py310, wgpu GPU compute) |
| `native/serverlist/`, `native/viewer/` | Small Rust tools: BeamMP Earth server list; browser viewer of earthd's output |
| `mod/beamng_earth/` | The game mod (Lua + Vue UI). Level `levels/beamng_earth`, extensions under `scripts/` + `lua/ge/extensions/beamngEarth`, Vue pages in `ui/ui-vue/mods/beamngEarth/` |
| `mod/tests/` | ~90 offline Lua unit tests (stubs, no game) + `.mjs` map-math tests |
| `server/tests/` | ~228 Python unittest modules |
| `tools/` | Build, test runners, probes, release helpers (see below) |
| `probes/` | Historical in-game probes (engine facts measured via MCP) |
| `beammp/` | BeamMP server-side plugin (`Server/BeamNGEarth/main.lua`) and host docs |
| `docs/` | `ARCHITECTURE.md` (engine facts + design), `V1_*_SPEC.md`, `ROADMAP.md`, `QUICKSTART.md`, research notes |
| `.claude/agent-rules.md` | The maintainer's standing rules — **read it first**; it is authoritative |
| `CHANGELOG.md`, `PR_CHANGELOG.md`, `SOURCES.md`, `CREDITS.md` | Release notes, PR-notes style, data sources, attribution |

## Architecture in three lines

- The game's Lua **cannot reach the internet** (sandbox); a local Python server (`earthd`) downloads open data
  (DEM, orthophotos, OSM/Overture, weather), builds tiles and streams them over localhost TCP.
- The game only ever sees a local ENU "bubble" around a moving anchor (floating origin, rebases ~8 km); the server
  keeps the real WGS84 Earth in float64. Near ground = one runtime TerrainBlock (1 km, 1 m); far = render-only
  binary `.cdae` tiles. Details and measured engine limits: `docs/ARCHITECTURE.md`.
- Heavy per-tile work lives in the Rust module (`native/earthfast`); Python is thin glue. The mod stays Lua.

## Running

- One-time: `python install.py` (or `py install.py` on Windows) → finds the BeamNG user folder, copies the mod,
  makes `.venv` with `server/requirements.txt`.
- Server only: `python start.py` (recommended flags) or `python server/earthd.py` (raw; flags in `server/README.md`).
- Everything: `python launcher.py` (`--no-game` for server only; `-- --verbose` passes through to earthd).
- Logs: `~/.cache/beamng-earth/logs/server.log` (Linux/Deck), `%LOCALAPPDATA%\beamng-earth\logs\server.log`.
- Cache: `~/.cache/beamng-earth` (default 20 GB cap). **Never point a scratch/test cache at the real one**
  (no symlinks/copies of its folders) — a real incident evicted 7 GB through symlinks. `BNGE_CACHE` overrides.
- Useful env vars: `BNGE_REVISIT`/`BNGE_REVISIT_MB` (revisit cache), `BNGE_ROADZ`, `BNGE_LIDAR_DECKS`,
  `BNE_APPROACH_LIDAR`, `BNGE_PEERS` (compute helpers), `BNGE_CATCHUP_LEND`, `BNGE_UPDATE_GOLDEN`,
  `BNGE_ROADGEO_REAL`, `BNGE_SKIP_DRIVE`.

## Tests and CI

- All server tests (parallel runner with logs): `python tools/run_tests.py [module ...]`
- One module: `cd server && python -m unittest tests.test_<name>` or `python -m unittest discover -s tests -t tests`
- Lua tests: `python tools/run_lua_tests.py` (needs `lupa`; or `luajit mod/tests/test_*.lua` individually)
- Full CI gate (what `quality.yml` runs on ubuntu+windows): `python tools/quality_gate.py`
  (server modules + Lua tests + `node --test` map math + `cargo test native/serverlist` + package build).
  Needs `pip install -r server/requirements.txt lupa==2.8` and `python tools/build_native.py` first.
- CI: `.github/workflows/quality.yml` (PRs) and `release.yml` (every push to `dev` publishes a
  `v<VERSION>.dev.<run>` pre-release; a `main` push whose `VERSION` has no release ships it).

## Building

- Rust module after touching `native/earthfast`: `python tools/build_native.py`
  (`--windows` cross-build with cargo-zigbuild; `--fast` dev profile — never commit a fast build; the marker file
  `server/earthfast.FAST-BUILD-do-not-commit` will show up).
  If a kernel signature changed: bump `m.add("VERSION", n)` in `lib.rs`, call through `fast.require(n, ...)`,
  and raise `fast.MIN_VERSION` in `server/fast.py` (current: 149).
- Player app: `python tools/build_app.py` (PyInstaller; run `tools/build_bootloader.py` first for the .exe).
  Release zip: `python tools/package.py`.

## Hard rules (from `.claude/agent-rules.md`, condensed; the file itself wins)

- **Rust, not Python.** New server logic goes into `native/earthfast`; Python is argument marshalling only.
  Never keep a Python twin of a Rust kernel. The mod stays Lua.
- **Branches:** work on `dev`; push `dev` after each tested batch. **Never** push `main`, bump `VERSION`, tag,
  or cut releases without an explicit go-ahead for that release. Nothing ships before in-game review.
- **Current checkout is on `main`:** switch to `dev` before doing any work unless told otherwise.
- **Look at the real place first** for any map/geometry fix: aerial photo (`tools/ortho_hi.py LAT LON HALF OUT.png`)
  and Mapillary (`tools/streetview.py fetch/run/compare`) before writing code; compare the game screenshot after.
- **Check open PRs first** (`gh pr list --state open`) before starting a feature; build on them, don't duplicate.
- **Outside code is untrusted:** read the entire diff of any PR/contribution for network calls, exec/eval,
  obfuscation, file/credential access, workflow/binary/lockfile changes before running or merging anything.
  Record `# security-reviewed: ...` in the commit. Suspicious → stop and notify. Never merge contributor PRs or
  run their code yourself; leave that to the user.
- **GitHub:** read issues/PRs freely; never comment, close, merge or review without the user's go (draft replies).
- **Verify before claiming:** check in-game with screenshots you actually look at, compare to real photos,
  report failures honestly. Units for the user: mph and miles.
- **Generalise:** no location-specific code; test places are benchmarks only.
- **Format bumps:** any change to grading plans or road facts output must bump `plancache.PLANS_FORMAT` /
  `roads.FACTS_FORMAT` in the same commit, then refresh golden data with `BNGE_UPDATE_GOLDEN=1`
  (`tests/test_bake_formats.py`). Pure speedups need byte-identical output proof to keep the numbers.
- **CHANGELOG/releases:** hedge wording, no "Known issues" section, end with the Discord line; release notes end
  with launch steps (`python tools/release_notes.py <version>`). Keep `README.md` and `docs/LAUNCH_STEPS.md` current.
- **Never `pkill -f`/`pgrep -f`** a pattern that matches your own command line; use brackets (`pgrep -f "[e]arthd"`).
- **Privacy:** the user's email never leaves the machine. Ask before any VirusTotal upload.

## Conventions worth knowing

- Commit style: one-line factual summaries, often with in-game measurements, PR/issue numbers and format numbers
  (see `git log`). The repo history is the spec of the current state; CHANGELOG is user-facing.
- Tests: server = `server/tests/test_*.py` (unittest, synthetic data; drive tests skip with `BNGE_SKIP_DRIVE=1`);
  mod = offline Lua with `mod/tests/stubs.lua` + `testlib.lua`. `tools/quality_gate.py` lists the must-pass core set.
- Rust crates: `native/earthfast` (cdylib, abi3), each with `cargo test`. `native/serverlist` is tested in CI.
- The server speaks length-prefixed UTF-8 JSON frames (protocol 4, `server/protocol.py`); mod/server capability
  negotiation happens in `hello` — check the flag table in `server/README.md` before touching message shapes.
- In-game work uses the game's built-in MCP server (`http://127.0.0.1:29292/mcp`, `tools/mcp.py`); after a game
  update run `tools/compat_check.py`. Vue UI files must be checked with
  `node .claude/devtools/vuecheck/check.js mod/beamng_earth/ui/ui-vue/mods/beamngEarth/*.vue` — the game compiles
  them at runtime and a template error only shows as a map that won't open.
- Server-side probes live in `server/` (`tileprobe.py`, `rampcheck.py`, `deckedges.py`, `floattrees.py`, ...) and
  dev scripts in `.claude/devtools/` (camshot, drives, shots); many assume the maintainer's PC/Deck hosts
  (10.0.0.5 / 10.0.0.21) and won't run elsewhere.

## Where to read next

1. `.claude/agent-rules.md` — full standing rules, tools and machine details.
2. `server/README.md` — every earthd flag, cache layout, protocol 4 messages.
3. `docs/ARCHITECTURE.md` — verified engine facts (ProceduralMesh limits, TerrainBlock quirks, rebase costs).
4. `README.md` + `docs/QUICKSTART.md` — current user-facing behavior (keep these updated with every change).
5. `docs/ROADMAP.md` — what's next.
