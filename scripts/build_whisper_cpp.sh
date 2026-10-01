#!/bin/sh
# Build whisper.cpp's whisper-server for Unsloth's GGUF dictation engine.
# Installs into the managed Unsloth home so the backend's binary discovery
# (core/inference/stt_ggml_sidecar.py::find_whisper_server_binary) picks it up:
#   <UNSLOTH_HOME>/whisper.cpp/build/bin/whisper-server          (master root)
#   <UNSLOTH_STUDIO_HOME>/whisper.cpp/build/bin/whisper-server   (custom home)
#   ~/.unsloth/whisper.cpp/build/bin/whisper-server              (default)
# Usage:
#   ./scripts/build_whisper_cpp.sh              # build the pinned tag
#   WHISPER_CPP_TAG=v1.9.0 ./scripts/build_whisper_cpp.sh
# Requires: git, cmake, a C/C++ toolchain (the same prerequisites as a
# llama.cpp source build). GPU backends are auto-detected by whisper.cpp's
# CMake (Metal on macOS; set GGML_CUDA=1 to force a CUDA build on Linux).
#
# The default source is the FiditeNemini fork at upstream v1.9.4 with its ggml/
# replaced by the ggml of the llama.cpp mix setup.sh builds (made by
# scripts/unsloth/make_ggml_tag.sh in the fork), so both share one ggml. The
# commit pin is checked after fetching; WHISPER_CPP_SOURCE / WHISPER_CPP_TAG
# override and drop it.

set -eu

_DEFAULT_WHISPER_CPP_SOURCE="https://github.com/FiditeNemini/whisper.cpp"
_DEFAULT_WHISPER_CPP_TAG="v1.9.4-ggml-b11160-mix-a6922cc"
_DEFAULT_WHISPER_CPP_COMMIT="8358c3d153d022028ef0a137a2167dc18032daf6"
WHISPER_CPP_PINNED_COMMIT=""
if [ -z "${WHISPER_CPP_SOURCE:-}" ] && [ -z "${WHISPER_CPP_TAG:-}" ]; then
    WHISPER_CPP_PINNED_COMMIT="$_DEFAULT_WHISPER_CPP_COMMIT"
fi
WHISPER_CPP_SOURCE="${WHISPER_CPP_SOURCE:-$_DEFAULT_WHISPER_CPP_SOURCE}"
WHISPER_CPP_TAG="${WHISPER_CPP_TAG:-$_DEFAULT_WHISPER_CPP_TAG}"

# Stripped and tilde-expanded before it can outrank anything, as setup.sh and the Python
# resolvers do: ${VAR:-} only treats the EMPTY string as unset, so a whitespace-only value
# would win here and name a relative "   /whisper.cpp" the backend never looks in.
_root_value() {
    _rv=$(printf '%s' "${1:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    case "$_rv" in
        "~") _rv="$HOME" ;;
        "~/"*) _rv="$HOME/${_rv#'~/'}" ;;
    esac
    printf '%s' "$_rv"
}

# UNSLOTH_HOME first: whisper.cpp is a SIBLING of studio/ under the master root, and the CLI
# exports UNSLOTH_STUDIO_HOME=<root>/studio beside it, so taking that one would install a level
# below where stt_ggml_sidecar._managed_whisper_cpp_dir() looks.
_STUDIO_HOME_ALIAS="${STUDIO_HOME:-}"   # read before the name below is reassigned
STUDIO_HOME="$(_root_value "${UNSLOTH_HOME:-}")"
[ -n "$STUDIO_HOME" ] || STUDIO_HOME="$(_root_value "${UNSLOTH_STUDIO_HOME:-}")"
[ -n "$STUDIO_HOME" ] || STUDIO_HOME="$(_root_value "$_STUDIO_HOME_ALIAS")"
CUSTOM_STUDIO_HOME=false
if [ -n "$STUDIO_HOME" ]; then
    CUSTOM_STUDIO_HOME=true
    INSTALL_DIR="$STUDIO_HOME/whisper.cpp"
else
    INSTALL_DIR="$HOME/.unsloth/whisper.cpp"
fi

command -v git >/dev/null 2>&1 || { echo "ERROR: git is required" >&2; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "ERROR: cmake is required" >&2; exit 1; }

# Same policy as studio/setup.sh's _assert_studio_owned_or_absent: never delete
# a directory under a custom Unsloth home unless Unsloth itself created it (the
# marker file below). Protects a user-managed whisper.cpp/src from rm -rf.
STUDIO_OWNED_MARKER=".unsloth-studio-owned"
if [ "$CUSTOM_STUDIO_HOME" = true ] && [ -e "$INSTALL_DIR" ] && \
   [ ! -f "$INSTALL_DIR/$STUDIO_OWNED_MARKER" ]; then
    echo "ERROR: $INSTALL_DIR already exists and is not marked as an Unsloth-owned whisper.cpp build tree." >&2
    echo "       Move it aside or choose an empty UNSLOTH_STUDIO_HOME before re-running." >&2
    exit 1
fi

echo "==> Building whisper.cpp ($WHISPER_CPP_TAG) into $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
: > "$INSTALL_DIR/$STUDIO_OWNED_MARKER"

# The commit the installed binary was built from; a matching pin skips the build.
SOURCE_STAMP="$INSTALL_DIR/build/bin/.whisper-source-commit"
if [ -n "$WHISPER_CPP_PINNED_COMMIT" ] && [ -x "$INSTALL_DIR/build/bin/whisper-server" ] && \
   [ "$(cat "$SOURCE_STAMP" 2>/dev/null || true)" = "$WHISPER_CPP_PINNED_COMMIT" ]; then
    echo "==> whisper-server already built at $WHISPER_CPP_TAG; skipping"
    exit 0
fi

if [ ! -d "$INSTALL_DIR/src/.git" ]; then
    rm -rf "$INSTALL_DIR/src"
    git clone --depth 1 --branch "$WHISPER_CPP_TAG" "$WHISPER_CPP_SOURCE" "$INSTALL_DIR/src"
else
    # An earlier install may have cloned another source (ggml-org before the fork).
    git -C "$INSTALL_DIR/src" remote set-url origin "$WHISPER_CPP_SOURCE"
    git -C "$INSTALL_DIR/src" fetch --depth 1 origin "$WHISPER_CPP_TAG"
    git -C "$INSTALL_DIR/src" checkout -q FETCH_HEAD
fi

BUILT_COMMIT="$(git -C "$INSTALL_DIR/src" rev-parse HEAD)"
if [ -n "$WHISPER_CPP_PINNED_COMMIT" ] && [ "$BUILT_COMMIT" != "$WHISPER_CPP_PINNED_COMMIT" ]; then
    echo "ERROR: $WHISPER_CPP_TAG is at $BUILT_COMMIT, not the pinned $WHISPER_CPP_PINNED_COMMIT -- refusing to build" >&2
    exit 1
fi

CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF"
if [ "${GGML_CUDA:-0}" = "1" ]; then
    CMAKE_FLAGS="$CMAKE_FLAGS -DGGML_CUDA=ON"
fi

# shellcheck disable=SC2086
cmake -S "$INSTALL_DIR/src" -B "$INSTALL_DIR/src/build" $CMAKE_FLAGS
NCPU="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
cmake --build "$INSTALL_DIR/src/build" --config Release --target whisper-server -j"$NCPU"

mkdir -p "$INSTALL_DIR/build/bin"
cp "$INSTALL_DIR/src/build/bin/whisper-server" "$INSTALL_DIR/build/bin/whisper-server"
printf '%s\n' "$BUILT_COMMIT" > "$SOURCE_STAMP"

echo "==> Installed $INSTALL_DIR/build/bin/whisper-server"
"$INSTALL_DIR/build/bin/whisper-server" --help >/dev/null 2>&1 && echo "==> Binary runs OK"
