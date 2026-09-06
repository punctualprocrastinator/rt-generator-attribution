"""Build the RT toolchain and turn exported RelBench databases into RT's native format.

Two stages, both straight out of the PluRel/RT repo, just orchestrated for 1000 databases instead
of one:

  stage 1  `rustler ... pre <db>`      -> ~/scratch/pre/<db>/{nodes,offsets,p2f_adj}.rkyv,
                                          text.json, column_index.json, table_info.json
  stage 2  text embedding              -> ~/scratch/pre/<db>/text_emb_<model>.bin

`embed_all` is the one place we deviate from calling the upstream script per database, and only in
orchestration: `python -m rt.embed <db>` reloads the SentenceTransformer every invocation (~10 s),
which would be ~3 h of pure model loading across 1000 databases. We load the identical model once
and encode each database's `text.json` in turn -- same model, same batching, same
`.astype(bfloat16).tofile()` output, so the bytes on disk are what the upstream script would write.
`verify_embeddings` re-checks the file sizes against `text.json` afterwards.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# --------------------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------------------- #
def setup_scratch(home: str | None = None) -> tuple[Path, Path]:
    """Create `~/scratch/{relbench,pre}` (the layout hardcoded in rt/ and rustler/)."""
    h = Path(home or os.environ["HOME"])
    rel = h / "scratch" / "relbench"
    pre = h / "scratch" / "pre"
    rel.mkdir(parents=True, exist_ok=True)
    pre.mkdir(parents=True, exist_ok=True)
    return rel, pre


def raise_fd_limit(target: int = 200_000) -> tuple[int, int]:
    """Raise RLIMIT_NOFILE.

    RT's Rust sampler memory-maps THREE files per task (nodes, text embeddings, p2f adjacency) and
    does NOT deduplicate databases across tasks -- so `num_tasks * 3` file descriptors are held
    open at once. With thousands of synthetic tasks the default soft limit (often 1024) is hit
    immediately, and the failure surfaces as an opaque `Os { code: 24 }` panic inside the sampler.
    """
    try:
        import resource
    except ImportError:  # Windows
        return (-1, -1)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = min(target, hard) if hard != resource.RLIM_INFINITY else target
    if soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)


def stage_local(
    src: str | Path, dest: str | Path = "/content/source_stage", jobs: int = 16
) -> Path:
    """Copy a Drive-resident source tree to local disk. **Opt-in, and usually not worth it.**

    Conversion reads every file exactly once, so for a SINGLE run staging buys nothing -- it just
    moves the same Drive cost up front, and `shutil.copytree` does it one file at a time. It only
    pays off when you expect to convert repeatedly (tuning task selection, re-running after a fix),
    where the second and later passes then read from local disk.

    Copies are parallelised because Drive FUSE is latency-bound rather than bandwidth-bound: many
    concurrent small reads are far faster than a serial walk.
    """
    src, dest = Path(src), Path(dest)
    if not src.is_dir():
        raise FileNotFoundError(f"source not found: {src}")
    if dest.exists() and any(dest.iterdir()):
        n = sum(1 for _ in dest.rglob("*") if _.is_file())
        print(f"[stage] reusing {dest} ({n:,} files already staged)")
        return dest

    files = [p for p in src.rglob("*") if p.is_file()]
    dest.mkdir(parents=True, exist_ok=True)
    for d in {p.parent for p in files}:
        (dest / d.relative_to(src)).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[stage] copying {len(files):,} files {src} -> {dest} with {jobs} workers ...",
          flush=True)

    def one(p: Path):
        shutil.copy2(p, dest / p.relative_to(src))
        return p.stat().st_size

    done = 0
    total = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for sz in ex.map(one, files):
            done += 1
            total += sz
            if done % 2000 == 0:
                print(f"    {done:,}/{len(files):,} files, {total/1e6:,.0f} MB, "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
    print(f"[stage] {done:,} files, {total/1e6:,.0f} MB in {(time.time()-t0)/60:.1f} min -> {dest}")
    return dest


def check_gpu() -> dict:
    """Report GPU capability. RT trains in bfloat16 and compiles FlexAttention kernels, both of
    which need compute capability >= 8.0 (Ampere). A T4 (7.5) has no bf16 tensor cores."""
    import torch

    info = {"cuda": torch.cuda.is_available(), "name": None, "capability": None, "bf16_ok": False}
    if info["cuda"]:
        info["name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info["capability"] = f"{cap[0]}.{cap[1]}"
        info["bf16_ok"] = cap[0] >= 8
    return info


# --------------------------------------------------------------------------------------- #
# stage 0: build the Rust sampler
# --------------------------------------------------------------------------------------- #
def _explain(output: str) -> str:
    """Append an actionable hint for build failures we have already diagnosed once."""
    hints = []
    if "E0658" in output and "`let` expressions" in output:
        hints.append(
            f"rustler's source uses a let-chain (src/fly.rs), stabilized in Rust 1.88, but this "
            f"toolchain is older. Set prep.RUST_TOOLCHAIN to >= 1.88.0 (currently "
            f"{RUST_TOOLCHAIN!r}) and rerun."
        )
    if "E0512" in output or "ethnum" in output:
        hints.append(
            "`ethnum 1.5.0` does not compile on newer toolchains (std changed TryFromIntError's "
            "layout). The toolchain is too NEW for the pinned dependency tree: lower "
            "prep.RUST_TOOLCHAIN, or run `cargo update -p ethnum` in the rustler dir."
        )
    if "signal: 9" in output or "SIGKILL" in output or "Killed" in output:
        hints.append(
            "A compiler/linker process was killed -- almost always out of memory. Rebuild with "
            "fewer parallel jobs: prep.build_rustler(REPO, jobs=2)."
        )
    if "Couldn't find a virtualenv" in output:
        hints.append(
            "`maturin develop` needs an active virtualenv. Use the wheel or cargo-cdylib path "
            "(build_rustler does this automatically)."
        )
    return ("\n--- hint ---\n" + "\n".join(f"* {h}" for h in hints)) if hints else ""


def _run(cmd, *, cwd=None, env=None, what="command", check=True, quiet=False, stream=False):
    """Run a command, capturing output, and on failure raise with the actual error text.

    Build failures here are otherwise invisible: a bare `subprocess.run(check=True)` raises
    CalledProcessError with no stderr attached, which turns a one-line toolchain problem into an
    opaque traceback.

    `stream=True` also echoes progress as it happens. Cargo builds 307 crates here and can run for
    tens of minutes; capturing silently makes a working build look like a hang.
    """
    args = [str(c) for c in cmd] if isinstance(cmd, (list, tuple)) else cmd
    shown = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)

    if stream:
        import collections

        p = subprocess.Popen(
            args,
            cwd=str(cwd) if cwd else None,
            env=env,
            shell=isinstance(cmd, str),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        tail_buf: collections.deque = collections.deque(maxlen=200)
        t0, n = time.time(), 0
        for line in p.stdout:
            tail_buf.append(line)
            n += 1
            # cargo emits one "Compiling <crate>" per crate: a natural progress bar
            if line.lstrip().startswith(("Compiling", "Building", "Finished", "error", "warning: b")):
                print(f"    [{(time.time()-t0)/60:5.1f}m] {line.rstrip()[:110]}", flush=True)
        p.wait()
        out = "".join(tail_buf)
        if p.returncode != 0 and check:
            raise RuntimeError(
                f"{what} failed (exit {p.returncode}):\n  $ {shown}\n"
                f"--- output (last 40 lines) ---\n"
                + "\n".join(out.strip().splitlines()[-40:])
                + _explain(out)
            )
        print(f"    [{(time.time()-t0)/60:5.1f}m] {what} done ({n} lines)", flush=True)

        class _R:
            returncode = p.returncode
            stdout = out
            stderr = ""

        return _R()

    r = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        shell=isinstance(cmd, str),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 and check:
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        tail = out.strip().splitlines()
        raise RuntimeError(
            f"{what} failed (exit {r.returncode}):\n  $ {shown}\n"
            f"--- output (last 40 lines) ---\n"
            + "\n".join(tail[-40:])
            + _explain(out)
        )
    if not quiet and r.stdout:
        print(r.stdout.strip()[-2000:])
    return r


def rust_version() -> tuple[int, int] | None:
    """(major, minor) of the rustc on PATH, or None if there isn't one."""
    if not shutil.which("rustc"):
        return None
    r = subprocess.run(["rustc", "--version"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        parts = r.stdout.split()[1].split(".")
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


# Building rustler needs a Rust toolchain inside a WINDOW, and both ends bite:
#
#   floor   1.88  rustler's own source uses a let-chain (`if let Some(x) = ... && cond`) at
#                 src/fly.rs:571. Let-chains stabilized in 1.88; below that:
#                     error[E0658]: `let` expressions in this position are unstable
#                 (Edition 2024 itself only needs 1.85, so the manifest understates the floor.)
#   ceiling   -   the committed Cargo.lock pins an early-2025 dependency tree (polars 0.46,
#                 pyo3 0.24). `ethnum 1.5.0` transmutes `()` into `std::num::TryFromIntError`,
#                 a ZST when that crate shipped; newer std changed its layout, so a current
#                 toolchain gives:
#                     error[E0512]: cannot transmute between types of different sizes
#                 (observed on 1.97).
#
# So `--default-toolchain stable` breaks on a 2026 image, and an older pin breaks on the source.
# 1.88 is the lowest version that satisfies the floor, which keeps it furthest from the ceiling.
# If a future toolchain still trips ethnum, `_cargo_retry` bumps that one crate and retries.
MIN_RUST = (1, 88)
RUST_TOOLCHAIN = "1.88.0"

#: Bumped whenever a source patch changes what the `pre` binary does. A cached binary carrying a
#: different tag is rejected by `restore_binary`, so a stale build cannot silently mask a fix.
#: Defined here because `restore_binary` uses it as a default argument, evaluated at import.
TOOLCHAIN_TAG = "relbench-v2-drops-1"


def ensure_rust(
    toolchain: str | None = RUST_TOOLCHAIN, min_version: tuple[int, int] = MIN_RUST
) -> tuple[int, int]:
    """Guarantee a Rust toolchain that can actually build rustler.

    Both bounds matter. Too old (< 1.85) fails with `feature 'edition2024' is required`; too new
    fails inside pinned dependencies. Pass `toolchain=None` to just use/att install `stable`.
    """
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.is_dir() and str(cargo_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{cargo_bin}:{os.environ.get('PATH', '')}"

    want = toolchain or "stable"
    ver = rust_version()

    if not shutil.which("rustup"):
        print(f"[rust] installing rustup + {want} ...")
        _run(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y "
            f"--default-toolchain {want} --profile minimal --no-modify-path",
            what="rustup install",
            quiet=True,
        )
        os.environ["PATH"] = f"{cargo_bin}:{os.environ.get('PATH', '')}"
    elif toolchain and ver != _parse_toolchain(toolchain):
        print(f"[rust] pinning toolchain to {toolchain} (found rustc {ver})")
        _run(["rustup", "toolchain", "install", toolchain, "--profile", "minimal"],
             what=f"rustup toolchain install {toolchain}", quiet=True)
        _run(["rustup", "default", toolchain], what="rustup default", quiet=True)
    elif not ver:
        _run(["rustup", "default", want], what="rustup default", quiet=True)

    ver = rust_version()
    if not ver or ver < min_version:
        raise RuntimeError(
            f"rustc is {ver} after install, need >= {min_version[0]}.{min_version[1]}. "
            f"Install manually: `curl https://sh.rustup.rs -sSf | sh -s -- -y "
            f"--default-toolchain {want}` and restart the runtime."
        )
    print(f"[rust] rustc {ver[0]}.{ver[1]} ready (pinned: {toolchain or 'stable'})")
    return ver


def _parse_toolchain(name: str) -> tuple[int, int] | None:
    try:
        parts = name.split("-")[0].split(".")
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


def _rust_env() -> dict:
    """Bake the interpreter's lib dir into the rpath -- the rustler binary links libpython
    through pyo3 and otherwise fails to find it at run time (same trick the upstream
    examples/inference/pipeline.py uses)."""
    libdir = sysconfig.get_config_var("LIBDIR") or ""
    flags = f"{os.environ.get('RUSTFLAGS', '')} -C link-args=-Wl,-rpath,{libdir}".strip()
    return os.environ | {
        "RUSTFLAGS": flags,
        "LD_LIBRARY_PATH": f"{libdir}:{os.environ.get('LD_LIBRARY_PATH', '')}",
    }


def restore_binary(
    src_dir: str | Path,
    dest: str | Path = "/content/rustler_bin",
    *,
    require_tag: str | None = TOOLCHAIN_TAG,
) -> Path | None:
    """Reuse a `pre` binary cached by an earlier session, skipping the whole cargo build.

    A cache written before the current source patches is REFUSED rather than reused: it would run
    the unpatched `pre` and reproduce the very panic the patch fixes, with nothing on screen to
    suggest the fix had not taken effect.
    """
    src_dir = Path(src_dir)
    if require_tag is not None:
        tag_file = src_dir / "PATCH_TAG"
        have = tag_file.read_text().strip() if tag_file.exists() else None
        if have != require_tag:
            print(
                f"[prep] cached `pre` binary is stale (tag {have!r}, need {require_tag!r}) "
                f"-- rebuilding so the source patches take effect"
            )
            return None
    for cand in ("rustler", "rustler.exe"):
        src = src_dir / cand
        if not src.exists():
            continue
        out = Path(dest)
        out.mkdir(parents=True, exist_ok=True)
        binp = out / cand
        shutil.copy(src, binp)
        try:
            binp.chmod(0o755)
            r = subprocess.run([str(binp), "--help"], capture_output=True, text=True, timeout=60)
            if r.returncode not in (0, 2):  # clap exits 2 on --help for some configurations
                raise RuntimeError(r.stderr[-400:])
        except Exception as e:  # noqa: BLE001 - a stale/incompatible binary just means rebuild
            print(f"[toolchain] cached binary unusable ({type(e).__name__}: {e}); rebuilding")
            return None
        print(f"[toolchain] reusing cached `pre` binary from {src} -- skipping the cargo build")
        return binp
    return None


def build_rustler(
    repo: str | Path, jobs: int | None = None, ext_dir: str | Path = "/content"
) -> Path:
    """Build the `rustler` python extension (for training) and CLI binary (for preprocessing).

    Two artifacts come out of one crate:

      * a **cdylib** imported as `rustler` by `rt.data` -- built with `pyo3/extension-module`
        (does NOT link libpython);
      * a **binary** providing the `pre` subcommand -- links libpython through pyo3, so the
        interpreter's lib dir is baked into its rpath.

    The extension is built with maturin when possible, but the crate carries BOTH a lib and a bin
    target and `maturin develop` additionally insists on an active virtualenv, neither of which
    holds on a bare Colab interpreter. So there is a maturin-free fallback that asks cargo for the
    cdylib directly and drops the `.so` on `sys.path` -- the same compiled object, without
    maturin's packaging step.

    Returns the path to the CLI binary.
    """
    ensure_rust()
    rustler = Path(repo) / "rustler"
    if not (rustler / "Cargo.toml").exists():
        raise FileNotFoundError(f"no rustler crate at {rustler}")
    env = _rust_env()
    print(f"[build] crate: {rustler}")
    print("[build] 307 crates incl. the whole polars + arrow stack, release mode: expect roughly "
          "10-20 min on a high-CPU runtime and 30-45 min on a 2-vCPU one. Progress is streamed "
          "below; `save_toolchain` caches the result so later sessions skip this entirely.")

    ext_ok = _build_extension(rustler, env, ext_dir)

    # ---- CLI binary with the `pre` subcommand ----
    cmd = ["cargo", "build", "--release", "--bin", "rustler"]
    if jobs:
        cmd += ["-j", str(jobs)]
    print("[build] cargo build --release --bin rustler")
    _cargo_retry(
        lambda: _run(cmd, cwd=rustler, env=env, what="cargo build --bin rustler", stream=True),
        rustler,
        env,
    )

    binary = rustler / "target" / "release" / "rustler"
    if not binary.exists():
        binary = binary.with_suffix(".exe")
    if not binary.exists():
        raise FileNotFoundError(f"rustler binary not found under {rustler / 'target' / 'release'}")

    # ---- prove the extension actually imports; that is the thing training depends on ----
    import importlib

    importlib.invalidate_caches()
    try:
        importlib.import_module("rustler")
    except ImportError as e:
        raise RuntimeError(
            f"the `rustler` python extension is not importable after building "
            f"(extension build path: {ext_ok}): {e}"
        ) from e
    print(f"[build] OK -- extension importable, binary at {binary}")
    return binary


def _cargo_retry(fn, rustler: Path, env: dict):
    """Run a cargo/maturin build; on a known dependency incompatibility, remedy it and retry once.

    `ethnum 1.5.0` (pulled in by polars) fails to compile on newer toolchains with E0512. If we
    still hit it, bumping that one crate within its semver range is a smaller, safer change than
    unpinning the whole lockfile.
    """
    try:
        return fn()
    except RuntimeError as e:
        if "E0512" not in str(e) and "ethnum" not in str(e):
            raise
        print("\n[build] hit the ethnum/E0512 incompatibility -- `cargo update -p ethnum`, retrying\n")
        try:
            _run(["cargo", "update", "-p", "ethnum"], cwd=rustler, env=env,
                 what="cargo update -p ethnum", check=False, quiet=True)
        except OSError as remedy_err:  # cargo missing/unrunnable -- keep the ORIGINAL error visible
            raise e from remedy_err
        return fn()


def _build_extension(rustler: Path, env: dict, ext_dir: str | Path) -> str:
    """Build the pyo3 cdylib and make it importable. Returns which strategy succeeded."""
    if not shutil.which("maturin"):
        _run([sys.executable, "-m", "pip", "install", "-q", "maturin"], what="pip install maturin",
             quiet=True)

    # strategy 1: maturin wheel + pip install (no virtualenv needed, unlike `maturin develop`)
    try:
        print("[build] maturin build --release")
        _cargo_retry(
            lambda: _run(
                ["maturin", "build", "--release", "-i", sys.executable,
                 "--compatibility", "linux"],
                cwd=rustler,
                env=env,
                what="maturin build",
                quiet=True,
            ),
            rustler,
            env,
        )
        wheels = sorted(
            (rustler / "target" / "wheels").glob("rustler-*.whl"), key=lambda p: p.stat().st_mtime
        )
        if not wheels:
            raise RuntimeError("maturin reported success but produced no wheel")
        _run(
            [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", str(wheels[-1])],
            what="pip install rustler wheel",
            quiet=True,
        )
        print(f"[build] extension installed from {wheels[-1].name}")
        return "maturin-wheel"
    except RuntimeError as e:
        print(f"\n[build] maturin path failed, falling back to plain cargo.\n{e}\n")

    # strategy 2: cargo builds the cdylib directly; copy the .so onto sys.path.
    # `pyo3/extension-module` keeps it from linking libpython, which is what makes it importable.
    print("[build] cargo rustc --release --lib --crate-type cdylib")
    _cargo_retry(
        lambda: _run(
            [
                "cargo", "rustc", "--release", "--lib",
                "--crate-type", "cdylib",
                "--features", "pyo3/extension-module",
            ],
            cwd=rustler,
            env=env,
            what="cargo rustc --lib",
            quiet=True,
        ),
        rustler,
        env,
    )
    built = None
    for cand in ("librustler.so", "rustler.so", "librustler.dylib", "rustler.dll"):
        p = rustler / "target" / "release" / cand
        if p.exists():
            built = p
            break
    if built is None:
        raise FileNotFoundError(
            f"cargo produced no cdylib in {rustler / 'target' / 'release'} "
            f"(looked for librustler.so / rustler.so)"
        )

    out_dir = Path(ext_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / ("rustler.pyd" if built.suffix == ".dll" else "rustler.so")
    shutil.copy(built, dest)
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    print(f"[build] extension copied {built.name} -> {dest}")
    return "cargo-cdylib"


def diagnose_toolchain() -> dict:
    """Report the build toolchain -- run this first when a build fails."""
    info = {}
    for name, cmd in [
        ("rustc", ["rustc", "--version"]),
        ("cargo", ["cargo", "--version"]),
        ("maturin", ["maturin", "--version"]),
        ("cc", ["cc", "--version"]),
    ]:
        exe = shutil.which(cmd[0])
        if not exe:
            info[name] = "NOT FOUND"
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        info[name] = (r.stdout or r.stderr).strip().splitlines()[0] if r.returncode == 0 else "error"
    info["python"] = sys.version.split()[0]
    info["min_rust_required"] = f"{MIN_RUST[0]}.{MIN_RUST[1]} (crate is edition 2024)"
    for k, v in info.items():
        print(f"  {k:20s} {v}")
    return info


# --------------------------------------------------------------------------------------- #
# stage 1: rustler pre
# --------------------------------------------------------------------------------------- #
SOURCE_MARKER = "rt_icl_source.json"


def _marker(path: Path) -> str | None:
    f = path / SOURCE_MARKER
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text()).get("source")
    except (json.JSONDecodeError, OSError):
        return None


def pre_done(db_name: str, home: str | None = None) -> bool:
    """True if `<db_name>` is preprocessed AND corresponds to the CURRENT export of that name.

    Database names are positional (`<gen>-synthetic-00037`), so the source a name refers to changes
    whenever the kept set changes -- e.g. dropping databases over the FK limit renumbers everything
    after the first drop. Resuming on the name alone would then reuse another database's
    preprocessed data under the new name: at best a crash deep in training ("Column ... not found in
    column_index.json"), at worst silent training on mismatched columns. So the identity of the
    source is recorded at conversion and compared here.
    """
    h = Path(home or os.environ["HOME"])
    p = h / "scratch" / "pre" / db_name
    if not all((p / f).exists() for f in ("table_info.json", "nodes.rkyv", "offsets.rkyv")):
        return False
    want = _marker(h / "scratch" / "relbench" / db_name)
    if want is None:  # no export to compare against (e.g. a pretraining session) -> trust the files
        return True
    return _marker(p) == want


def backfill_markers(db_names: list[str], home: str | None = None, verbose: bool = True) -> int:
    """Stamp unmarked preprocessed dirs that PROVABLY match the current export.

    Preprocessed data produced before provenance markers existed is indistinguishable from stale
    data by marker alone, so `pre_done` conservatively calls it stale and re-runs. That is correct
    but wasteful when the data is in fact current. Here we check it directly: every column of the
    current export must appear in the preprocessed `column_index.json`. If it does, the pre output
    describes this export and can be stamped; if not, it is left unmarked and will be rebuilt.
    """
    import pyarrow.parquet as pq

    h = Path(home or os.environ["HOME"])
    stamped = skipped = 0
    for db in db_names:
        pre = h / "scratch" / "pre" / db
        rel = h / "scratch" / "relbench" / db
        if _marker(pre) is not None or not (pre / "column_index.json").exists():
            continue
        want = _marker(rel)
        if want is None or not (rel / "db").is_dir():
            continue
        try:
            have = set(json.loads((pre / "column_index.json").read_text()))
            expected = {
                f"{c} of {f.stem}"
                for f in (rel / "db").glob("*.parquet")
                for c in pq.ParquetFile(f).schema_arrow.names
            }
        except Exception:  # noqa: BLE001 - unreadable means "rebuild it"
            skipped += 1
            continue
        if expected and expected <= have:
            (pre / SOURCE_MARKER).write_text(json.dumps({"source": want, "db_name": db}))
            stamped += 1
        else:
            skipped += 1
    if verbose and (stamped or skipped):
        print(f"[backfill] stamped {stamped} preprocessed database(s) that match the current "
              f"export; {skipped} did not match and will be rebuilt")
    return stamped


def stamp_pre_sources(db_names: list[str], home: str | None = None) -> int:
    """Copy each export's source marker into its preprocessed dir. Returns how many were stamped."""
    h = Path(home or os.environ["HOME"])
    n = 0
    for db in db_names:
        src = h / "scratch" / "relbench" / db / SOURCE_MARKER
        dst = h / "scratch" / "pre" / db
        if src.exists() and dst.is_dir():
            shutil.copy(src, dst / SOURCE_MARKER)
            n += 1
    return n


def verify_pre_matches_manifest(
    manifest: dict, home: str | None = None, check_columns: bool = True
) -> list[str]:
    """Check the preprocessed data on disk really is the corpus this manifest describes.

    Two failure modes, both caused by positional names outliving the data they named:
      * the preprocessed database came from a different source (marker mismatch);
      * a task's target column does not exist in that database (`column_index.json`).

    Returns a list of problems; empty means safe to train.
    """
    h = Path(home or os.environ["HOME"])
    problems = []
    for d in manifest["databases"]:
        db = d["db_name"]
        p = h / "scratch" / "pre" / db
        if not p.is_dir():
            problems.append(f"{db}: not preprocessed")
            continue
        got = _marker(p)
        if got is not None and got != d.get("source"):
            problems.append(f"{db}: preprocessed from {got!r}, manifest says {d.get('source')!r}")
            continue
        if not check_columns:
            continue
        ci = p / "column_index.json"
        if not ci.exists():
            problems.append(f"{db}: no column_index.json")
            continue
        try:
            cols = set(json.loads(ci.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            problems.append(f"{db}: unreadable column_index.json ({e})")
            continue
        for t in d["tasks"]:
            key = f"{t['target']} of {t['table']}"
            if key not in cols:
                problems.append(f"{db}: task column {key!r} missing from column_index.json")
    return problems


# `pre.rs` hardcodes per-database fixups written against RelBench **v1**, each ending in
# `.unwrap()`. The benchmark installs relbench 2.1.2 for the v2 datasets, where some of those
# columns no longer exist -- and "drop this column" on a column that is already gone raises
# ColumnNotFound and panics. Both sites below mean "make sure this column is not present", which a
# missing column already satisfies, so guarding them is exactly equivalent where the column exists
# and correct where it does not.
_RUST_PATCHES = [
    (
        '            if table_name == "posts" {\n'
        '                df = df.drop("AcceptedAnswerId").unwrap();\n'
        '            }',
        '            if table_name == "posts" {\n'
        '                // rt_icl: absent in RelBench v2, where this drop is already satisfied\n'
        '                if df.column("AcceptedAnswerId").is_ok() {\n'
        '                    df = df.drop("AcceptedAnswerId").unwrap();\n'
        '                }\n'
        '            }',
    ),
    (
        '                df = df.drop("index").unwrap();\n'
        '                df = cast_col_to_bool(df, "target").unwrap();',
        '                // rt_icl: v2 task tables may not carry the positional index column\n'
        '                if df.column("index").is_ok() {\n'
        '                    df = df.drop("index").unwrap();\n'
        '                }\n'
        '                df = cast_col_to_bool(df, "target").unwrap();',
    ),
]


def patch_rustler_relbench_v2(repo: str | Path) -> bool:
    """Make `pre.rs`'s hardcoded v1 column drops tolerate RelBench v2. Idempotent.

    Returns True if the source changed, which means any cached `pre` binary is stale and the
    caller must rebuild. Raises if an anchor is missing rather than leaving the panic in place.
    """
    p = Path(repo) / "rustler" / "src" / "pre.rs"
    src = p.read_text(encoding="utf-8")
    if "rt_icl: absent in RelBench v2" in src:
        return False
    for old, new in _RUST_PATCHES:
        if old not in src:
            raise RuntimeError(
                f"could not find a v1-fixup block in {p} -- rustler changed upstream; update "
                f"rt_icl.prep._RUST_PATCHES. Missing:\n{old}"
            )
        src = src.replace(old, new, 1)
    p.write_text(src, encoding="utf-8")
    print(f"[prep] patched {p}: RelBench v2 column drops made conditional ({len(_RUST_PATCHES)} sites)")
    return True


def system_ram_gb() -> float | None:
    """Total system RAM in GB, or None if it cannot be read."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def pre_failure_text(r) -> str:
    """Explain a `pre` failure, including the case where it produced no error message at all.

    `pre` writes its `dbg!` output to **stderr** and its progress to **stdout**, so a run that is
    KILLED rather than crashing leaves a stderr full of debug lines and no message -- which reads
    like a crash with a meaningless tail. Naming the exit status turns that into a diagnosis.
    """
    rc = r.returncode
    sig = -rc if rc < 0 else (rc - 128 if rc > 128 else None)
    out = [f"exit code {rc}" + (f" (killed by signal {sig})" if sig is not None else "")]
    if sig == 9:
        ram = system_ram_gb()
        out.append(
            "  SIGKILL with no panic = the process was KILLED, not crashed. On Colab that is\n"
            "  essentially always the out-of-memory killer: `pre` holds the whole database plus\n"
            "  every task table in RAM at once."
            + (f"\n  This runtime has {ram:.0f} GB." if ram else "")
            + "\n  Fix: Runtime -> Change runtime type -> High-RAM, and prepare that database\n"
            "  in a session of its own. rel-amazon is the one that needs it."
        )
    elif sig is not None:
        out.append(f"  killed by signal {sig} without panicking -- not a Rust error")
    elif "panicked at" not in (r.stderr or ""):
        out.append("  non-zero exit with no Rust panic in stderr")
    out.append(f"--- stderr tail ---\n{(r.stderr or '')[-1200:]}")
    out.append(f"--- stdout tail ---\n{(r.stdout or '')[-600:]}")
    return "\n".join(out)


def run_pre(
    db_names: list[str],
    binary: str | Path,
    *,
    home: str | None = None,
    jobs: int = 4,
    log_dir: str | Path | None = None,
    progress_every: int = 50,
) -> dict:
    """Run `<binary> pre <db>` for each database. Resumable, parallel, collects failures."""
    binary = str(binary)
    env = _rust_env()
    todo = [d for d in db_names if not pre_done(d, home)]
    print(f"[pre] {len(db_names)} databases, {len(db_names)-len(todo)} already done, {len(todo)} to do")
    log_dir = Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    failures, done = {}, 0
    t0 = time.time()

    h = Path(home or os.environ["HOME"])

    def one(db: str):
        r = subprocess.run(
            [binary, "pre", db], capture_output=True, text=True, env=env, cwd=Path(binary).parent
        )
        if r.returncode != 0:
            if log_dir:
                (log_dir / f"pre_{db}.log").write_text((r.stdout or "") + (r.stderr or ""))
            return db, pre_failure_text(r)
        # Stamp provenance immediately: `pre_done` compares this against the export's marker, so
        # without it every database would look stale the instant it finished preprocessing.
        src = h / "scratch" / "relbench" / db / SOURCE_MARKER
        if src.exists():
            shutil.copy(src, h / "scratch" / "pre" / db / SOURCE_MARKER)
        return db, None

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for db, err in ex.map(one, todo):
            done += 1
            if err:
                failures[db] = err
            if done % progress_every == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                print(
                    f"[pre] {done}/{len(todo)}  {el/60:.1f} min  "
                    f"{rate:.1f} db/s  failures={len(failures)}",
                    flush=True,
                )
    if failures:
        print(f"[pre] {len(failures)} FAILED, e.g.:")
        for db, err in list(failures.items())[:3]:
            print(f"  --- {db} ---\n{err[-600:]}")
    return failures


# --------------------------------------------------------------------------------------- #
# stage 2: text embeddings (model loaded once for all databases)
# --------------------------------------------------------------------------------------- #
def embed_all(
    db_names: list[str],
    *,
    embedding_model: str = "all-MiniLM-L12-v2",
    home: str | None = None,
    batch_size: int = 8192,
    device: str | None = None,
    progress_every: int = 50,
    chunk: int = 250_000,
) -> dict:
    """Embed every database's `text.json`, reusing one SentenceTransformer.

    Same model, same float32->bfloat16 cast, same raw `.tofile()` layout and same text order as
    `python -m rt.embed <db>`; only the model load is amortized and the result is streamed to disk
    in `chunk`-sized pieces instead of being held whole in RAM.

    Chunking does not change the values: MiniLM masks padding in both attention and mean pooling,
    so a sentence's embedding does not depend on which other sentences share its batch. (Strictly,
    differing tensor shapes can perturb the last bit of a float accumulation, so this is
    numerically equivalent rather than bit-identical -- irrelevant at bfloat16's ~3 decimal
    digits, and `verify_embeddings` still checks the file has exactly one vector per text.)
    """
    import numpy as np
    import torch
    from ml_dtypes import bfloat16
    from sentence_transformers import SentenceTransformer

    h = Path(home or os.environ["HOME"])
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        f"sentence-transformers/{embedding_model}",
        device=device,
        model_kwargs={"dtype": torch.bfloat16 if device == "cuda" else torch.float32},
    )

    failures, done, t0 = {}, 0, time.time()
    for db in db_names:
        pre = h / "scratch" / "pre" / db
        out = pre / f"text_emb_{embedding_model}.bin"
        tj = pre / "text.json"
        done += 1
        if out.exists() and out.stat().st_size > 0:
            continue
        if not tj.exists():
            failures[db] = "no text.json (pre stage did not run)"
            continue
        try:
            texts = json.loads(tj.read_text())
            if not texts:
                # rustler mmaps this file; a zero-byte mmap fails, so never write an empty one
                texts = [""]
            # Stream to disk in chunks. Encoding everything in one call holds the whole (N, 384)
            # float32 result in RAM -- ~7.7 GB at 5M texts, and `np.stack` on an already-2D array
            # silently doubled that with a full copy. The big RelBench databases have text
            # vocabularies that size, which is what was killing this step.
            tmp = out.with_name(out.name + ".part")
            n_done = 0
            with open(tmp, "wb") as fh:
                for i in range(0, len(texts), chunk):
                    emb = model.encode(
                        texts[i:i + chunk], batch_size=batch_size,
                        convert_to_numpy=True, show_progress_bar=False,
                    )
                    np.asarray(emb).astype(bfloat16).tofile(fh)
                    n_done += len(emb)
                    if len(texts) > chunk:
                        print(f"[embed] {db}: {n_done:,}/{len(texts):,} texts", flush=True)
            # Atomic: a half-written file must never look like a finished one to a later session.
            tmp.replace(out)
        except Exception as e:  # noqa: BLE001 - report and continue over 1000 databases
            failures[db] = f"{type(e).__name__}: {e}"
            try:
                out.with_name(out.name + ".part").unlink(missing_ok=True)
            except OSError:
                pass
        if done % progress_every == 0 or done == len(db_names):
            print(
                f"[embed] {done}/{len(db_names)}  {(time.time()-t0)/60:.1f} min  "
                f"failures={len(failures)}",
                flush=True,
            )
    if failures:
        print(f"[embed] {len(failures)} FAILED, e.g. {list(failures.items())[:3]}")
    return failures


def verify_embeddings(
    db_names: list[str], embedding_model: str = "all-MiniLM-L12-v2", d_text: int = 384,
    home: str | None = None,
) -> list[str]:
    """Check every embedding file has exactly len(text.json) * d_text bfloat16 values."""
    h = Path(home or os.environ["HOME"])
    bad = []
    for db in db_names:
        pre = h / "scratch" / "pre" / db
        f = pre / f"text_emb_{embedding_model}.bin"
        tj = pre / "text.json"
        if not f.exists() or not tj.exists():
            bad.append(f"{db}: missing files")
            continue
        n = max(len(json.loads(tj.read_text())), 1)
        expect = n * d_text * 2  # bfloat16 = 2 bytes
        got = f.stat().st_size
        if got != expect:
            bad.append(f"{db}: {got} bytes, expected {expect} ({n} texts x {d_text} x 2)")
    return bad


# --------------------------------------------------------------------------------------- #
# handing preprocessed data between sessions
# --------------------------------------------------------------------------------------- #
def _tar_create_cli(pre: Path, names: list[str], dest: Path, compress: bool) -> bool:
    """Archive with the `tar` binary (much faster than Python's tarfile on multi-GB corpora).

    Returns False if it is unavailable or refuses the paths (on Windows, GNU tar reads a
    drive-letter path as a remote host), so the caller can fall back to the portable path.
    """
    if not shutil.which("tar"):
        return False
    import tempfile

    listfile = Path(tempfile.gettempdir()) / "rt_pre_list.txt"
    listfile.write_text("\n".join(names))
    r = subprocess.run(
        ["tar", "-C", str(pre), "-czf" if compress else "-cf", str(dest), "-T", str(listfile)],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        return True
    print(f"[archive] tar CLI unavailable for these paths, using Python tarfile "
          f"({(r.stderr or '').strip()[:120]})")
    return False


def _tar_extract_cli(src: Path, pre: Path) -> bool:
    if not shutil.which("tar"):
        return False
    r = subprocess.run(["tar", "-C", str(pre), "-xf", str(src)], capture_output=True, text=True)
    if r.returncode == 0:
        return True
    print(f"[restore] tar CLI unavailable for these paths, using Python tarfile "
          f"({(r.stderr or '').strip()[:120]})")
    return False


def archive_pre(
    db_names: list[str], dest: str | Path, *, home: str | None = None, compress: bool = False
) -> Path:
    """Tar the preprocessed databases so a later session can train without re-converting.

    `~/scratch/pre` is ephemeral, so splitting conversion and pretraining across sessions means
    carrying it over. One tar beats copying thousands of small files to Drive by a wide margin.
    Only the named databases are archived, so several generators can share one Drive folder.
    Compression is off by default: the bulk is already-packed binary (`.rkyv`, bfloat16
    embeddings) and gzip mostly burns CPU for a few percent.
    """
    h = Path(home or os.environ["HOME"])
    pre = h / "scratch" / "pre"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    present = [d for d in db_names if (pre / d).is_dir()]
    missing = len(db_names) - len(present)
    if not present:
        raise FileNotFoundError(f"none of the {len(db_names)} databases exist under {pre}")
    if missing:
        print(f"[archive] {missing} database(s) not found under {pre}; archiving {len(present)}")

    t0 = time.time()
    print(f"[archive] {len(present)} databases -> {dest} ...", flush=True)
    if not _tar_create_cli(pre, present, dest, compress):
        import tarfile

        with tarfile.open(dest, "w:gz" if compress else "w") as tf:
            for name in present:
                tf.add(pre / name, arcname=name)
    gb = dest.stat().st_size / 1e9
    print(f"[archive] {gb:.2f} GB in {(time.time()-t0)/60:.1f} min -> {dest}")
    return dest


def restore_pre(src: str | Path, *, home: str | None = None) -> int:
    """Unpack an `archive_pre` tar back into `~/scratch/pre`. Returns the database count."""
    h = Path(home or os.environ["HOME"])
    pre = h / "scratch" / "pre"
    pre.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(
            f"no preprocessed archive at {src} -- run the conversion notebook for this generator "
            f"first, or point PRE_ARCHIVE at the tar it wrote"
        )
    t0 = time.time()
    print(f"[restore] unpacking {src} ({src.stat().st_size/1e9:.2f} GB) -> {pre} ...", flush=True)
    if not _tar_extract_cli(src, pre):
        import tarfile

        with tarfile.open(src) as tf:
            tf.extractall(pre)
    n = sum(1 for d in pre.iterdir() if d.is_dir() and (d / "table_info.json").exists())
    print(f"[restore] {n} databases ready in {(time.time()-t0)/60:.1f} min")
    return n


def save_toolchain(repo: str | Path, dest_dir: str | Path) -> Path | None:
    """Stash the compiled rustler artifact on Drive so the next session skips the Rust build.

    The training session still imports `rustler` (rt.data does), and rebuilding it costs ~10
    minutes of GPU-session time for no reason.
    """
    repo, dest_dir = Path(repo), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Record which source patches this build carries, so a later session can tell whether the
    # cached binary is still the right one instead of assuming it is.
    (dest_dir / "PATCH_TAG").write_text(TOOLCHAIN_TAG)
    saved = None
    # the `pre` CLI: this is what the conversion notebook needs, and caching it means a rerun
    # never pays the cargo build again
    for cand in ("rustler", "rustler.exe"):
        binp = repo / "rustler" / "target" / "release" / cand
        if binp.exists() and binp.is_file():
            out = dest_dir / cand
            shutil.copy(binp, out)
            out.chmod(0o755)
            print(f"[toolchain] saved `pre` binary -> {out} ({out.stat().st_size/1e6:.0f} MB)")
            saved = out
            break

    wheels = sorted(
        (repo / "rustler" / "target" / "wheels").glob("rustler-*.whl"),
        key=lambda p: p.stat().st_mtime,
    )
    if wheels:
        out = dest_dir / wheels[-1].name
        shutil.copy(wheels[-1], out)
        print(f"[toolchain] saved wheel -> {out}")
        return out
    if saved:
        return saved
    for cand in ("librustler.so", "rustler.so"):
        so = repo / "rustler" / "target" / "release" / cand
        if so.exists():
            out = dest_dir / "rustler.so"
            shutil.copy(so, out)
            print(f"[toolchain] saved extension -> {out}")
            return out
    print("[toolchain] nothing to save (no wheel or .so found)")
    return None


def install_toolchain(src_dir: str | Path, ext_dir: str | Path = "/content") -> bool:
    """Install a stashed rustler artifact. Returns True if `import rustler` then works."""
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        return False
    wheels = sorted(src_dir.glob("rustler-*.whl"), key=lambda p: p.stat().st_mtime)
    try:
        if wheels:
            _run(
                [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", str(wheels[-1])],
                what="pip install stashed rustler wheel",
                quiet=True,
            )
        elif (src_dir / "rustler.so").exists():
            out = Path(ext_dir)
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_dir / "rustler.so", out / "rustler.so")
            if str(out) not in sys.path:
                sys.path.insert(0, str(out))
        else:
            return False
        import importlib

        importlib.invalidate_caches()
        importlib.import_module("rustler")
        print(f"[toolchain] reused prebuilt rustler from {src_dir} (skipped the Rust build)")
        return True
    except Exception as e:  # noqa: BLE001 - any failure just means we build from source
        print(f"[toolchain] could not reuse prebuilt rustler ({type(e).__name__}: {e}); building")
        return False


# --------------------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------------------- #
def write_manifest(path: str | Path, payload: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p


def read_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
