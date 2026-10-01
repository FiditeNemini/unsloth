# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Contracts for the update popup's release-notes preview, on the frontend side.

Studio no longer fetches or parses release notes (utils/release_notes.py answers "no notes"
without a lookup; see studio/backend/tests/test_no_update_checks.py). What stays pinned here is
the popup, banner and preview code and the overlay layout they share, which the desktop
updater path and the shared helpers in tests/_shared still rely on."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "studio/backend"
FRONTEND = REPO / "studio/frontend/src"
MODULE = BACKEND / "utils/release_notes.py"
PANEL = FRONTEND / "components/update/release-notes-panel.tsx"
NOTES_LAYOUT = FRONTEND / "components/update/update-notes-layout.ts"
NOTES_HOOK = FRONTEND / "hooks/use-release-notes.ts"
PREVIEW = FRONTEND / "lib/release-notes-preview.ts"
CODE_SPANS = FRONTEND / "lib/markdown-code-spans.ts"
LINKS = FRONTEND / "lib/release-body-links.ts"
LIST_COLUMNS = FRONTEND / "lib/markdown-list-columns.ts"
INLINE_COMMENTS = FRONTEND / "lib/markdown-inline-comments.ts"
WEB_BANNER = FRONTEND / "components/web/update-banner.tsx"
TAURI_BANNER = FRONTEND / "components/tauri/update-banner.tsx"


# An apostrophe in JSX text is prose, not the start of a string: "We're ready"
# in a banner's copy would otherwise run a scanner to the next apostrophe or off
# the end of the file, and a copy edit would fail these tests. The frontend is
# formatted to double quotes, so nothing here is delimited with `'`.
# A set of characters, not a string: `"" in '"`'` is true for a substring, and
# a trailing comma leaves an empty argument to test.
_QUOTES = frozenset('"`')

_CALL = re.compile(r"[A-Za-z_$][\w$]*\(")

_IMPORTANT = re.compile(r"^!|!$")


def _split_variants(token: str) -> tuple[tuple[str, ...], str]:
    """A Tailwind class token as (variants, utility), split on top-level colons.

    Depth-aware, because an arbitrary value may carry its own brackets and its
    own colon: `has-[[data-slot=update-release-notes]]:min-h-[calc(...)]`.
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in token:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
        if char == ":" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    # `!min-h-0` and `min-h-0!` are `min-h-0`, at a weight that beats the floor.
    # Left as written, an important rule would slip past a prohibition on the
    # plain one while overriding it. The frontend already writes them, in
    # `app/routes/__root.tsx` among others.
    return tuple(parts[:-1]), _IMPORTANT.sub("", parts[-1])


def _tokens(source: str) -> list[tuple[tuple[str, ...], str]]:
    """Every class token in `source`, as (variants, utility)."""
    return [_split_variants(token) for token in re.findall(r"""[^\s"'`]+""", source)]


def _applies(source: str, utility: str, *variants: str) -> bool:
    """Is `utility` written anywhere in `source` under at least `variants`?

    Token-wise, not as a substring. These assertions used to name a run of
    classes verbatim, so gating a rule (`max-[383px]:has-[...]:min-h-[calc]`)
    or inserting an unrelated one beside it read as the rule being gone, and
    #10229 went red on four of them while every rule it named was still there.
    A utility is the last top-level segment, so a longer utility that merely
    ends with this one does not count, and variant order is Tailwind's business
    rather than this test's.

    Name no variants and this asks whether the utility is there under any gate
    or none, which is what a check for a rule's *absence* wants. A check that a
    rule is in force needs `_only_under`.
    """
    for token_variants, token_utility in _tokens(source):
        if token_utility == utility and set(variants) <= set(token_variants):
            return True
    return False


def _only_under(source: str, utility: str, *variants: str) -> bool:
    """Is `utility` written at least once, and every time under exactly these?

    What a positive layout guarantee needs, and neither half of it is a subset
    test. An extra gate narrows when the rule is in force, so a floor written
    `md:has-[...]:min-h-[...]` leaves every width from 384px to the `md`
    breakpoint with none, the narrow override having stopped at 383px. A
    second, ungated copy widens it the other way, back to the reserved empty
    height the gate was added to stop. Both leave the utility present, so both
    pass an existence check.

    Named with no variants, this is "written, and never gated": an ungated
    `shrink-0` is the whole guarantee when there are no notes, and
    `max-[383px]:shrink-0` would satisfy an existence check while leaving the
    card squeezable at every width but one.
    """
    found = False
    for token_variants, token_utility in _tokens(source):
        if token_utility != utility:
            continue
        if set(token_variants) != set(variants):
            return False
        found = True
    return found


def _class_const(source: str, name: str) -> str:
    """The class string of an exported `const NAME = "..."`.

    Comments blanked and anchored on the `export`, like the banner extractors:
    an old declaration left commented out above the live one would otherwise
    answer for it, and a stale copy that still reads correctly is exactly how a
    regression in the live one goes unnoticed.
    """
    match = re.search(
        rf'export const {re.escape(name)}\s*=\s*\n?\s*"([^"]*)"', _without_comments(source)
    )
    assert match, f"{name} is not an exported class constant"
    return match.group(1)


# Moved to tests/_shared/jsx_tags.py so tests/studio/test_overlay_layering.py reads rails with
# the same bracket- and literal-aware scanner instead of a private rfind("<")/find(">") pair.
from jsx_tags import (  # noqa: E402
    opening_tag as _opening_tag,
    skip_literal as _skip_literal,
    without_comments as _without_comments,
)


def _class_on_testid(source: str, testid: str) -> str:
    """The literal class string of the element carrying `data-testid=testid`.

    Anchored on the attribute that names the element rather than on a run of
    its classes. An anchor built from classes cannot survive one of them being
    inserted or reordered, which is the failure this file is being fixed for,
    and it fails by raising rather than by reporting a missing rule.
    """
    clean = _without_comments(source)
    start, end = _opening_tag(clean, clean.index(f'data-testid="{testid}"'))
    return _class_value(clean[start:end], testid)


def _class_value(tag: str, what: str) -> str:
    """Every class named by the `className` of `tag`, joined.

    A literal today. Wrapping one in the `cn()` this file already uses renders
    the same DOM, so it has to read the same rather than being skipped, which
    would have taken the next element's classes instead and reported every rule
    on this one as missing.
    """
    key = "className="
    assert key in tag, f"{what} carries no className"
    at = tag.index(key) + len(key)
    if tag[at] == '"':
        return tag[at + 1 : _skip_literal(tag, at) - 1]
    assert tag[at] == "{", f"{what}'s className is neither a literal nor an expression"
    return _always_rendered(tag[at + 1 : _balanced(tag, at) - 1])


def _arguments(call: str) -> list[str]:
    """The arguments of a call, split on its top-level commas."""
    at = call.index("(")
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    index = at
    while index < len(call):
        char = call[index]
        if char in _QUOTES:
            end = _skip_literal(call, index)
            current.append(call[index:end])
            index = end
            continue
        if char in "([{":
            depth += 1
            if depth == 1:
                index += 1
                continue
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                break
        elif char == "," and depth == 1:
            parts.append("".join(current))
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    parts.append("".join(current))
    return parts


def _always_rendered(expression: str) -> str:
    """The classes `expression` renders in every state, joined.

    Only an argument that is a bare literal. `cn(open && "x")` renders `x`
    sometimes and a rule the card must always carry is not satisfied by
    sometimes; written against a constant, `cn(false && "x")` renders it never,
    while the text of the class sits there in the file either way. Reading the
    literals out of the whole expression would call all three the same.
    """
    text = expression.strip()
    if text[:1] in _QUOTES and _skip_literal(text, 0) == len(text):
        return text[1:-1]
    if "(" not in text:
        return ""
    literals = []
    for argument in _arguments(text):
        part = argument.strip()
        if part[:1] in _QUOTES and _skip_literal(part, 0) == len(part):
            literals.append(part[1:-1])
        elif _CALL.match(part) and _balanced(part, part.index("(")) == len(part):
            # Grouping the arguments in a nested `cn()` renders the same
            # classes, so it has to read the same rather than as none at all.
            literals.append(_always_rendered(part))
    return " ".join(literals)


def _balanced(source: str, at: int) -> int:
    """The index just past the bracket group opening at `at`."""
    depth = 0
    index = at
    while index < len(source):
        char = source[index]
        if char in _QUOTES:
            index = _skip_literal(source, index)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise AssertionError("unbalanced brackets")


def _assert_classes(class_string: str, *rules: str) -> None:
    """Every one of `rules` is its own class in `class_string`."""
    present = set(class_string.split())
    missing = [rule for rule in rules if rule not in present]
    assert not missing, f"{missing} missing from {class_string!r}"


# The scanners are the frontend half of the contract the parser implements, so they are
# run rather than read. Node strips the types and nothing imports a package: no install.
_TS_ALIAS = re.compile(r'"@/lib/([a-z-]+)"')
_TS_RUNNER = """
import { resolveReleaseBodyLinks } from "./release-body-links.ts";
import { releaseNotesPreview } from "./release-notes-preview.ts";

const chunks: Buffer[] = [];
process.stdin.on("data", (chunk: Buffer) => chunks.push(chunk));
process.stdin.on("end", () => {
  const markdown = Buffer.concat(chunks).toString("utf8");
  const result =
    process.argv[2] === "links"
      ? resolveReleaseBodyLinks(markdown)
      : releaseNotesPreview(markdown);
  process.stdout.write(JSON.stringify(result));
});
"""

@dataclass(frozen = True)
class Section:
    """A heading and the lines under it, to the next heading of any level."""

    version: str
    heading: str
    body: str


def sections(module, text: str) -> list[Section]:
    """Every document-level heading in `text`, with the lines beneath it.

    The shipped scanner decides what a heading is; this only groups its events.
    """
    found: list[Section] = []
    bodies: list[list[str]] = []
    for event in module.scan_blocks(text):
        if isinstance(event, module.Heading):
            # A setext heading is the paragraph above it, already collected.
            if bodies and event.retract:
                del bodies[-1][len(bodies[-1]) - event.retract :]
            title = event.title.strip()
            found.append(Section(version = title.split()[0] if title else "", heading = title, body = ""))
            bodies.append([])
            continue
        if bodies:
            bodies[-1].append(event.line)
    return [
        Section(version = entry.version, heading = entry.heading, body = "\n".join(body).strip())
        for entry, body in zip(found, bodies)
    ]


@pytest.fixture(scope = "module")
def notes_module():
    sys.path.insert(0, str(BACKEND))
    try:
        from utils import release_notes
    finally:
        sys.path.pop(0)
    release_notes.reset_release_notes_cache()
    yield release_notes
    release_notes.reset_release_notes_cache()


def test_no_changelog_file_is_packaged_or_read():
    """The releases are the only source now. A file left in the packaging would
    be a second one, editable in a checkout and stale in a wheel."""
    assert not (REPO / "CHANGELOG.md").exists()
    assert not (REPO / "_changelog_build.py").exists()
    for name in ("pyproject.toml", "build.sh", ".gitignore"):
        assert "CHANGELOG.md" not in (REPO / name).read_text(encoding = "utf-8")
    source = MODULE.read_text(encoding = "utf-8")
    # "Full Changelog" is the footer line it strips; a file is what must be gone.
    assert "CHANGELOG.md" not in source and "changelog.py" not in source


def test_preview_keeps_identifier_underscores():
    """UNSLOTH_DISABLE_UPDATE_CHECK must not render as UNSLOTHDISABLEUPDATECHECK."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "BOLD_UNDERSCORE" in src and "ITALIC_UNDERSCORE" in src
    assert "parkCodeSpans" in src, "code spans are parked so their underscores survive"
    assert "const EMPHASIS" not in src, "the blanket emphasis strip is gone"


def test_panel_prefers_the_page_the_notes_came_from():
    """The page the notes came from wins over the caller's URL and the changelog."""
    src = PANEL.read_text(encoding = "utf-8")
    assert "notes?.htmlUrl ?? releaseNotesUrl ?? notes?.releaseNotesUrl" in src


def test_preview_keeps_comparison_operators():
    """The tag strip must keep the operators in "Support Python <3.15 and >3.9"."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "/<\\/?[a-zA-Z][^>]*>/g" in src, "tag strip must require a name character"


def test_preview_hides_commented_out_notes():
    """Unpublished notes inside <!-- --> are not rendered, so not previewed."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "stripCommentSpans" in src and "COMMENT_OPEN" in src


def test_hook_treats_a_reported_failure_as_retryable():
    src = NOTES_HOOK.read_text(encoding = "utf-8")
    assert "next.error !== null" in src


def test_hook_never_returns_another_versions_notes():
    """State still describes the previous version until the effect runs."""
    src = NOTES_HOOK.read_text(encoding = "utf-8")
    assert "notes.version === version" in src
    assert "load(version, true)" in src, "retry must ask the backend to bypass its cache"


def test_desktop_notes_link_to_the_release_page_on_every_platform():
    """manualReleaseUrl is Linux-package only; the rest need the release page."""
    hook = (FRONTEND / "hooks/use-tauri-update.ts").read_text(encoding = "utf-8")
    assert "const releasePageUrl = info ?" in hook
    banner = TAURI_BANNER.read_text(encoding = "utf-8")
    assert "releaseNotesUrl={releasePageUrl ?? manualReleaseUrl}" in banner
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    assert "releasePageUrl={update.releasePageUrl}" in provider


def test_preview_matches_how_markdown_renders_prose_and_links():
    """Three rendering mismatches the preview must not reintroduce: wrapped
    paragraphs split into fragments, autolinks eaten as tags, a lead cut short."""
    src = PREVIEW.read_text(encoding = "utf-8")
    # Contiguous prose lines accumulate and flush at a paragraph boundary.
    assert "collector.paragraph = collector.paragraph" in src
    # <https://x> renders as link text, so it is not a tag.
    assert "AUTOLINK" in src
    # "e.g. GGUF" is not a sentence boundary.
    assert "ABBREVIATIONS" in src and "INITIAL" in src


def test_preview_treats_code_as_literal():
    """Inside a code span, and inside an indented code block, Markdown renders
    the text literally, so the preview must not transform or promote it."""
    src = PREVIEW.read_text(encoding = "utf-8")
    # Code spans are parked before any other inline transformation.
    park = src.index("parkCodeSpans(markdown")
    assert park < src.index("stripHtmlTags(\n    parked")
    # A "- cmd" line inside an indented code block is not a headline bullet.
    assert "INDENTED_CODE_INDENT" in src


def test_desktop_updater_metadata_maps_published_field_names():
    """latest.json publishes Tauri's `notes`/`pub_date`; the manual Linux path
    must read those, not `body`/`date`, or its release notes are always empty."""
    rust = (REPO / "studio/src-tauri/src/desktop_update_policy.rs").read_text(encoding = "utf-8")
    assert 'alias = "body"' in rust and "notes: Option<String>" in rust
    assert 'alias = "date"' in rust and "pub_date: Option<String>" in rust
    assert "body: metadata.notes" in rust and "date: metadata.pub_date" in rust
    workflow = (REPO / ".github/workflows/release-desktop.yml").read_text(encoding = "utf-8")
    assert "'notes': notes," in workflow, "workflow no longer publishes `notes`"


def test_backend_exposes_release_notes_route():
    src = (BACKEND / "main.py").read_text(encoding = "utf-8")
    assert '@app.get("/api/studio/release-notes")' in src
    assert "is_supported_version_query" in src


def test_panel_is_scrollable_and_shows_only_the_stripped_notes():
    src = PANEL.read_text(encoding = "utf-8")
    assert "overflow-y-auto" in src, "release notes must scroll inside the popup"
    assert "max-h-" in src, "the scroller needs a bounded height"
    # latest.json's `notes` is install boilerplate, the same every release.
    assert "fallbackMarkdown" not in src
    assert "notes?.matched ? notes.markdown : null" in src


def test_notes_surface_is_borderless_and_lifts_in_dark_mode():
    src = PANEL.read_text(encoding = "utf-8")
    layout = NOTES_LAYOUT.read_text(encoding = "utf-8")
    assert "border border-border" not in src, "the notes box is a fill, not a bordered box"
    # Lighter than the card behind it, rather than a darker inset. #11459 respelled the
    # shorthand as an rgb() whose alpha scales with --contrast-wash-gain, which is the same
    # 0.06 white at the default gain of 1, so read the white and the amount rather than one
    # spelling. A darker inset, or a different amount, still fails. The class also has to end
    # where the match does, and start where it starts. Anything else and Tailwind reads a
    # different candidate than the one named here: `-broken` names no utility at all, `/50`
    # is a different lift, `:broken` is a variant on nothing, and a `hover:` in front paints
    # the lift only under the pointer instead of on the dark surface.
    assert re.search(
        r"(?:(?<=[\s\"'`])|^)"
        r"dark:bg-(?:white/\[0\.06\]"
        r"|\[rgb\(255_255_255_/_calc\(0\.06\*var\(--contrast-wash-gain,\s*1\)\)\)\])"
        r"(?=[\s\"'`]|$)",
        layout,
    ), "the dark notes surface is no longer a 0.06 white lift"
    # Streamdown's mt-6 clips the first heading against the scroller edge.
    assert "[&>*>*:first-child]:mt-0" in src
    # Shared utility: thumb hidden until the notes are hovered.
    assert "hover-scrollbar" in src
    # Streamdown renders code at text-sm, twice this panel's body size.
    assert "[&_code]:text-[0.92em]" in src


def test_hook_discards_notes_for_a_different_version():
    src = NOTES_HOOK.read_text(encoding = "utf-8")
    assert "notesVersion !== version" in src


def test_collapsed_panel_previews_the_top_bullets():
    """Collapsed popups show the headline changes without an extra click."""
    preview = PREVIEW.read_text(encoding = "utf-8")
    assert "RELEASE_NOTES_PREVIEW_ITEMS = 4" in preview
    # Wrapped bullets join into one item, or a preview ends mid-sentence.
    assert "collectBullets" in preview and "flush" in preview
    # Nested list items are detail, not headline changes.
    assert "NESTED_INDENT_TOLERANCE" in preview
    # Tag stripping repeats: one pass turns `<<b>b>` back into a live tag.
    assert "while (out !== previous)" in preview

    panel = PANEL.read_text(encoding = "utf-8")
    assert "releaseNotesPreview" in panel
    assert 'data-testid="update-release-notes-summary"' in panel
    # Fetched when the popup appears: the collapsed preview needs them too.
    assert "enabled: true" in panel


def test_preview_highlights_the_leading_sentence():
    """Each bullet leads with its headline sentence, emphasised over the rest."""
    preview = PREVIEW.read_text(encoding = "utf-8")
    assert "splitLeadSentence" in preview
    # A period inside "unsloth.ai" or "e.g." must not read as a break.
    assert "SENTENCE_BREAK" in preview and "(?=" in preview

    panel = PANEL.read_text(encoding = "utf-8")
    layout = NOTES_LAYOUT.read_text(encoding = "utf-8")
    assert "UPDATE_NOTES_LEAD_CLASS" in panel
    assert '"font-medium text-foreground"' in layout
    assert "item.rest" in panel


def _max_widths(source: str) -> set[str]:
    """Every `max-w-[...]` in *source*, read at the default UI scale.

    #11648 wrapped these lengths in `calc(Npx*var(--ui-space-scale,1))` so they follow the interface size, which is
    Npx at the default scale. Read back that way, the contract stays about the width rather than its spelling.
    """
    return {_SCALED_PX.sub(r"\1", width) for width in re.findall(r"max-w-\[([^\]\s\"]+)\]", source)}


_SCALED_PX = re.compile(r"calc\((\d+(?:\.\d+)?px)\*var\(--ui-space-scale,1\)\)")
# The notes width as written: `_max_widths` reads a bare 448px identically, so the scale is pinned raw.
_NOTES_WIDTH_SCALED = "max-w-[calc(448px*var(--ui-space-scale,1))]"


@pytest.mark.parametrize("banner", [WEB_BANNER, TAURI_BANNER])
def test_update_popups_share_the_notes_width(banner):
    """Every update popup uses the same width for its notes and action rows."""
    source = banner.read_text(encoding = "utf-8")
    assert "448px" in _max_widths(source)
    assert (
        _NOTES_WIDTH_SCALED in source and "max-w-[448px]" not in source
    ), "the notes width stopped scaling"
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    assert "400px" not in _max_widths(provider), "stack must not cap overlay width"
    llama = (FRONTEND / "components/llama-update-banner.tsx").read_text(encoding = "utf-8")
    assert "448px" in _max_widths(llama)
    assert (
        _NOTES_WIDTH_SCALED in llama and "max-w-[448px]" not in llama
    ), "the notes width stopped scaling"
    assert "400px" not in _max_widths(llama)


@pytest.mark.parametrize("banner", [WEB_BANNER, TAURI_BANNER])
def test_banners_toggle_inline_release_notes(banner):
    src = banner.read_text(encoding = "utf-8")
    assert "ReleaseNotesPanel" in src
    assert "Show release notes" in src and "Hide release notes" in src
    # Keyed by version, so a new offer cannot leave old notes on screen.
    assert "notesVersion" in src


@pytest.mark.parametrize(
    "banner,toggle,action",
    [
        (WEB_BANNER, "web-update-release-notes-toggle", "web-update-snooze-button"),
        (TAURI_BANNER, "tauri-update-release-notes-toggle", "Remind me later"),
    ],
)
def test_notes_toggle_shares_the_action_row(banner, toggle, action):
    """The toggle sits in the same row as the actions, not on its own line."""
    src = banner.read_text(encoding = "utf-8")
    row = src.index("mt-4 flex")
    assert row < src.index(toggle) < src.index(action)
    # Same type size as the actions beside it; nowrap keeps labels on one line.
    toggle_line = next(line for line in src.splitlines() if toggle in line)
    toggle_block = src[src.index("Button", row) : src.index(toggle_line)]
    assert "text-ui-13" in toggle_block and "whitespace-nowrap" in toggle_block


def test_preview_skips_raw_html_blocks():
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "stripRawHtml" in src
    # Anchored: only a line-leading tag opens a block, matching the parser.
    assert "/^ {0,3}<(pre|script|style|textarea)" in src


def test_preview_joins_an_indented_continuation_line():
    """Four spaces only start code outside a paragraph; inside one it is a wrap."""
    src = PREVIEW.read_text(encoding = "utf-8")
    # Measured from the line's container, so an item's own indent does not count.
    assert "!insideBlock && line.indent - line.column >= INDENTED_CODE_INDENT" in src
    # A fence indented into a list item is a block, not a wrapped line.
    assert "opensDeepFence" in src


def test_preview_code_spans_need_a_matching_closer():
    """A closer is a run of the same length, so the inner backticks survive."""
    src = CODE_SPANS.read_text(encoding = "utf-8")
    assert "candidate === ticks" in src, "a closer is a run of the same length"
    assert "stripPadding" in src, "one space of padding is dropped, as in Markdown"


def test_preview_skips_thematic_breaks():
    """`- - -` renders as a rule, so it must not take a preview slot."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "THEMATIC_BREAK" in src
    assert "THEMATIC_BREAK.test(visible)" in src


def test_preview_keeps_quoted_examples_out_of_the_headlines():
    """A quoted list is example output, not a change."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "quoted: boolean" in src
    assert "if (!line.quoted)" in src, "quoted bullets never become headlines"


def test_notes_panel_keeps_the_link_when_the_lookup_fails():
    """The release page can be reachable when the backend lookup is not."""
    src = PANEL.read_text(encoding = "utf-8")
    error_branch = src[src.index('if (state === "error")') :]
    retry = error_branch.index("update-release-notes-retry")
    assert error_branch.index("{link}") > retry, "link sits beside retry"


def test_hook_waits_for_the_desktop_auth_token():
    """A token not installed yet must not be recorded as a failed lookup."""
    src = NOTES_HOOK.read_text(encoding = "utf-8")
    assert "hasAuthToken()" in src and "AUTH_POLL_LIMIT" in src


def test_preview_skips_every_raw_block_form():
    """The extractor tracks the same block forms as the parser."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "RAW_BLOCKS" in src
    assert "CDATA" in src and "[A-Za-z]" in src


@pytest.mark.parametrize("banner", [WEB_BANNER, TAURI_BANNER])
def test_expanded_popup_fits_a_short_viewport(banner):
    """A window under roughly 430px used to push the card's title off screen."""
    panel = PANEL.read_text(encoding = "utf-8")
    # The notes region shrinks inside the capped card, so header and actions stay on screen.
    assert "min-h-0 flex-1" in panel, "notes height must follow the viewport"
    src = banner.read_text(encoding = "utf-8")
    assert "max-h-[calc(100dvh_-_2rem)]" in src, "card is the backstop on tiny viewports"


def test_relative_release_body_links_point_at_the_repository():
    """Repository-relative links would resolve against Unsloth's own origin."""
    src = LINKS.read_text(encoding = "utf-8")
    assert "https://github.com/unslothai/unsloth/blob/main/" in src
    assert "https://raw.githubusercontent.com/unslothai/unsloth/main/" in src
    # Absolute targets, fragments, fenced code and code spans stay untouched.
    assert "ABSOLUTE" in src and "codeSpans" in src and "FENCE" in src
    panel = PANEL.read_text(encoding = "utf-8")
    assert "resolveReleaseBodyLinks" in panel


@pytest.mark.parametrize("query", ["latest", "main", "not-a-version", "abc"])
def test_unparseable_versions_are_rejected(notes_module, query):
    """A query that cannot parse is a bad request, not an empty result."""
    assert notes_module.is_supported_version_query(query) is False


@pytest.mark.parametrize("query", ["2026.7.5", "v2026.7.5", "2026.07.5", "1.0.0rc1"])
def test_real_versions_are_still_accepted(notes_module, query):
    assert notes_module.is_supported_version_query(query) is True


def test_reference_style_images_resolve_to_the_raw_host():
    """An image needs the raw file: the blob URL is an HTML page."""
    src = LINKS.read_text(encoding = "utf-8")
    assert "IMAGE_REFERENCE" in src
    assert "imageLabels" in src


def test_collapsed_notes_surface_is_hidden_when_nothing_previews():
    """Notes that preview as nothing leave an empty strip, worse than none."""
    src = PANEL.read_text(encoding = "utf-8")
    assert "preview?.items.length === 0" in src


def test_code_spans_close_on_a_run_of_equal_length():
    """`a``b [x](y.md)` is one code span, so the link inside it is literal."""
    src = CODE_SPANS.read_text(encoding = "utf-8")
    assert "candidate === ticks" in src, "closer length must match the opener"
    # Shared, so the preview and the link resolver cannot drift apart.
    assert "markdown-code-spans" in PREVIEW.read_text(encoding = "utf-8")
    assert "markdown-code-spans" in LINKS.read_text(encoding = "utf-8")


def test_preview_decodes_entities_like_the_renderer():
    """Streamdown renders `AT&amp;T` as AT&T, so the raw entity must not show."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "NAMED_ENTITIES" in src and "decodeEntity" in src
    # Decoded before code spans are restored, so code keeps the literal text.
    assert src.index(".replace(ENTITY, decodeEntity)") < src.index(".replace(PARKED")


def test_release_notes_request_refreshes_an_expired_token():
    """A direct fetch cannot recover from a 401; authFetch refreshes first."""
    src = NOTES_HOOK.read_text(encoding = "utf-8")
    assert "authFetch(" in src
    assert "getAuthToken" not in src


def test_preview_handles_the_desktop_updater_line_endings():
    """CRLF used to hide fences and promote a code sample to a headline."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "LINE_ENDINGS" in src
    assert "LINE_ENDINGS" in LINKS.read_text(encoding = "utf-8")


def test_preview_renders_reference_links_as_text():
    """`[text][label]` renders as a link, so its raw markup must not show."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "LINK_REFERENCE" in src and "IMAGE_REFERENCE" in src
    # A definition line renders as nothing, so it is not a preview item.
    assert "DEFINITION" in src


def test_preview_treats_escaped_punctuation_as_literal():
    """`\\*not italic\\*` keeps its stars, and an escaped backtick opens no span."""
    assert "ESCAPE" in PREVIEW.read_text(encoding = "utf-8")
    assert "escaped(" in CODE_SPANS.read_text(encoding = "utf-8")


def test_link_resolver_skips_every_code_form():
    """Indented code and cross-line code spans render as code, so leave them."""
    src = LINKS.read_text(encoding = "utf-8")
    assert "INDENTED_CODE" in src
    # Spans are scanned over the whole document, not line by line.
    assert "codeSpans(masked)" in src
    # A definition cannot interrupt a paragraph.
    assert "definition.has(index)" in src


def test_badge_links_resolve_both_targets():
    """`[![alt](img)](link)`: the outer link needs a nested label to resolve."""
    assert "NESTED_LABEL" in LINKS.read_text(encoding = "utf-8")


def test_in_flight_requests_are_identified_not_just_versioned():
    """Two requests for one version could resolve out of order."""
    assert "requestIdRef" in NOTES_HOOK.read_text(encoding = "utf-8")


def test_notes_repair_the_shared_previews_width_reset():
    """MarkdownPreview clears max-width on descendants, so wide content escapes."""
    src = PANEL.read_text(encoding = "utf-8")
    assert "[&_img]:max-w-full" in src
    assert "[&_[data-streamdown=link-safety-modal]>*]:max-w-md" in src


@pytest.mark.parametrize("banner", [WEB_BANNER, TAURI_BANNER])
def test_only_the_notes_region_scrolls(banner):
    """The dismiss control sits inside the card, so the card must not scroll."""
    src = banner.read_text(encoding = "utf-8")
    surface = _card_surface(src)
    # The painted surface: capped, and a column, so the region inside it is the
    # one that scrolls.
    _assert_classes(surface, "flex", "max-h-[calc(100dvh_-_2rem)]", "flex-col")
    # Neither card scrolls. Asserted as the absence of a scrolling overflow
    # rather than as the presence of `overflow-hidden`, because those are two
    # different claims: the browser card now clips nothing at all, and reading
    # the clip as the no-scroll guarantee is what tied this contract to a
    # mechanism instead of to what it is for.
    assert not re.search(
        r"(?<![\w-])overflow-(?:y-)?(?:auto|scroll)(?![\w-])", surface
    ), "the card scrolls, so its dismiss control can leave the viewport"
    if banner == WEB_BANNER:
        # This surface floors itself at header + notes + actions, which is only
        # true while it declares neither of the two things that set a flex
        # item's automatic minimum size to zero. Either one back and the rail
        # squeezes the card until the action row is cut.
        for zeroes_the_floor in ("min-h-0", "overflow-hidden"):
            assert (
                zeroes_the_floor not in surface.split()
            ), f"{zeroes_the_floor} puts the browser card's floor back to nothing"
    else:
        _assert_classes(surface, "min-h-0", "overflow-hidden")
    layout = NOTES_LAYOUT.read_text(encoding = "utf-8")
    _assert_classes(
        _class_const(layout, "UPDATE_NOTES_ROOT_CLASS"),
        "flex",
        "min-h-0",
        "flex-1",
        "flex-col",
        "overflow-hidden",
    )
    panel = PANEL.read_text(encoding = "utf-8")
    assert "UPDATE_NOTES_EXPANDED_SCROLL_CLASS" in panel
    _assert_classes(
        _class_const(layout, "UPDATE_NOTES_EXPANDED_SCROLL_CLASS"),
        "max-h-64",
        "min-h-0",
        "flex-1",
        "overflow-y-auto",
    )
    # The collapsed summary scrolls too: without it the bullets were painted
    # over the row of buttons once the card's slot for them got small.
    _assert_classes(
        _class_on_testid(panel, "update-release-notes-summary"),
        "min-h-0",
        "flex-1",
        "space-y-1",
        "overflow-y-auto",
    )


def test_preview_follows_commonmark_paragraph_rules():
    """Only an ordered list starting at 1 may interrupt a paragraph, an
    unresolved reference keeps its brackets, and a quote owns its own."""
    src = " ".join(PREVIEW.read_text(encoding = "utf-8").split())
    assert "const interrupts = collector.current === null" in src
    assert "!collector.quotedParagraph;" in src
    assert "definedLabel" in src, "a reference only renders as text when defined"
    # A comment written mid-sentence hides its own line at most.
    assert "COMMENT_BLOCK_OPEN" in src


def test_link_resolver_leaves_raw_blocks_and_escapes_alone():
    src = LINKS.read_text(encoding = "utf-8")
    assert "RAW_HTML_OPEN" in src and "inRawHtml" in src
    assert "isEscaped(line, opener)" in src
    # A heading ends a paragraph, so a definition under one is a definition.
    assert "BLOCK_LINE.test(structure)" in src


def test_code_span_closers_ignore_backslashes():
    """Escapes are not processed inside a code span, so a run after one closes."""
    src = CODE_SPANS.read_text(encoding = "utf-8")
    # Counted over the whole module rather than from an exported wrapper: the
    # scanner has already moved above `codeSpans` once, and a slice anchored on
    # a wrapper reads as "no opener is escaped either" when that happens.
    calls = [
        " ".join(line.split())
        for line in src.splitlines()
        if "escaped(" in line and not line.lstrip().startswith("function escaped(")
    ]
    assert len(calls) == 1, f"only an opener can be escaped, called at {calls}"
    # And that one call guards the run that opens a span, not the one closing it.
    assert '!== "`" || escaped(' in calls[0]


# The desktop card's incompressible height, a fixed part plus a part that
# follows Settings > Appearance rather than one number measured at the default
# 15px: at the 20px maximum the action row wraps at every card width. The
# browser card used to carry the same pair of constants and now floors itself
# off its own content instead (see _assert_floors_itself); this one still names
# the height, and the same staleness is waiting for it.
_SCALED_FLOOR_TAURI = "min-h-[calc(117px+93px*var(--ui-font-scale,1))]"
_NARROW_FLOOR_TAURI = "min-h-[calc(24px+224px*var(--ui-font-scale,1))]"
_NARROW = "max-[383px]"
# A floor only has to exist while there are notes to give up, and gating it is
# what stopped the card reserving height it painted nothing into (#10229). So
# the guarantee under test is a pair: floored while the notes panel is there,
# and not squeezable at all while it is not.
_NOTES_GATE = "has-[[data-slot=update-release-notes]]"


_BANNER_ROOT = re.compile(r'data-testid="(?:web|tauri)-update-banner"')


def _unpositioned_branch(text: str) -> str:
    """The `: ...` arm of `positioned ? ... : ...`.

    The two arms are two different elements: `positioned` is the standalone
    banner, and the rail-facing card is the alternative. Reading both at once
    would let a rule move from the card to the standalone banner and still
    satisfy a check about the card. Split at the colon at bracket depth zero
    and outside any literal, so a Tailwind variant in the first arm
    (`dark:bg-card`) is not mistaken for the separator.

    Ternary nesting is counted, not just brackets. A ternary inside the first
    arm has a colon of its own at the same bracket depth, and taking that one
    returns the tail of the `positioned` arm as though it were the card, so
    every floor could be asserted against the wrong element.
    """
    index = text.index("?", text.index("positioned")) + 1
    pending = 1
    depth = 0
    while index < len(text):
        char = text[index]
        if char in _QUOTES:
            index = _skip_literal(text, index)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "?" and depth == 0 and text[index + 1 : index + 2] not in (".", "?"):
            pending += 1
        elif char == ":" and depth == 0 and text[index - 1] != "?":
            pending -= 1
            if pending == 0:
                return text[index + 1 :]
        index += 1
    raise AssertionError("the positioned card has no unpositioned branch")


def _card_slot(source: str) -> str:
    """The rail-facing root of an update card, comments stripped.

    Not one string literal: the root is a `cn()` of several, so an assertion
    anchored on the first of them cannot see the floor at all. Anchored on the
    `data-testid` that names the card, so no class has to keep its place for
    the root to be found, and narrowed to the branch the overlay rail actually
    renders. Comments go because both files name the very classes under test in
    prose beside them, and a rule that a comment can satisfy is not tested.
    """
    clean = _without_comments(source)
    match = _BANNER_ROOT.search(clean)
    assert match, "the update card has lost its data-testid"
    start, end = _opening_tag(clean, match.start())
    key = "className={cn("
    tag = clean[start:end]
    assert key in tag, "the card's root no longer builds its classes with cn()"
    # The classes it always renders, not the text of the branch: a floor put
    # behind a constant is still written in the file while reaching no DOM.
    return _always_rendered(_unpositioned_branch(clean[start + tag.index(key) : end]))


def _card_surface(source: str) -> str:
    """The painted surface: the class string of the card's first child.

    Bounded to that child rather than taken as the next literal `className` in
    the file, which is the dismiss button's the moment the surface writes its
    own classes through `cn()` instead.
    """
    clean = _without_comments(source)
    match = _BANNER_ROOT.search(clean)
    assert match, "the update card has lost its data-testid"
    _, end = _opening_tag(clean, match.start())
    child = clean.index("<", end)
    # A fragment emits no element, so it is not the surface and its `<` is not
    # the surface's. Wrapping the card in one changes no rendered class.
    while clean[child + 1] == ">":
        child = clean.index("<", child + 2)
    start, child_end = _opening_tag(clean, child + 1)
    return _class_value(clean[start:child_end], "the painted surface")


def _assert_floored(source: str, scaled: str, narrow: str, card: str) -> None:
    """The card keeps room for its header and buttons, in both of its states."""
    # Read off the rail-facing root, so a floor written on some inner box, or
    # on the standalone `positioned` banner, does not answer for this one.
    root = _card_slot(source)
    # `_only_under` and not an existence check, in all four: a floor that gains
    # a gate stops applying over part of its range, and one that gains an
    # ungated twin reserves the empty height the gate was added to stop.
    assert _only_under(
        root, scaled, _NOTES_GATE
    ), f"the {card} card's floor is fixed, ungated, or gated more than its notes"
    assert _only_under(
        root, narrow, _NOTES_GATE, _NARROW
    ), f"the {card} card's floor misses the narrow card's extra button row"
    # With no notes rendered there is no floor, so this is what holds the row.
    assert _only_under(root, "shrink-0"), f"the rail can squeeze the {card} card with no notes open"
    assert _only_under(
        root, "shrink", _NOTES_GATE
    ), f"the {card} card cannot give up its notes' height, so the rail clips its buttons"
    assert not _applies(
        root, "min-h-0"
    ), f"min-h-0 lets the rail squeeze the {card} card past its floor"


def _assert_floors_itself(source: str, card: str) -> None:
    """The card keeps room for its header and buttons without naming a height.

    The written-out floor it replaces was a constant against one type size, and
    #11458 made the spacing inside the card follow the interface font size too,
    so at the 20px setting the constant came to 209px while the content needed
    about 301px and the action row was cut. A measured floor cannot go stale
    that way, but it only exists while the surface declares neither `min-h-0`
    nor `overflow-hidden`: each of those sets a flex item's automatic minimum
    size to zero, which is what made a hand-written floor necessary at all.
    """
    root = _card_slot(source)
    surface = _card_surface(source)
    for zeroes_the_floor in ("min-h-0", "overflow-hidden"):
        assert (
            zeroes_the_floor not in surface.split()
        ), f"{zeroes_the_floor} leaves the {card} card with no floor at all"
    assert not re.search(
        r"(?<![\w-])min-h-\[", surface + root
    ), f"the {card} card names a height again, which goes stale at the next type size"
    # Unchanged from the written-floor days: the rail may only take the height
    # the notes are there to give up.
    assert _only_under(root, "shrink-0"), f"the rail can squeeze the {card} card with no notes open"
    assert _only_under(
        root, "shrink", _NOTES_GATE
    ), f"the {card} card cannot give up its notes' height, so the rail clips its buttons"
    assert not _applies(
        root, "min-h-0"
    ), f"min-h-0 lets the rail squeeze the {card} card past its floor"


RAIL_TESTID = "overlay-rail"


def _corner_rails(provider: str) -> list[str]:
    """The class strings of the bottom-right overlay rails.

    Anchored on ``data-testid`` rather than on a run of the rail's own classes. The old
    matcher spelled the corner INTO the pattern - `bottom-0 right-4` - so #11260 moving the
    rail flush to the edge (`right-0`, with the inset paid as inline px padding) made it
    match nothing, and four tests across two files failed at once while reporting a missing
    rail rather than a changed one. The corner is a CLAIM these tests make, so it belongs in
    an assertion, not in the thing that finds the element to assert about. Same reasoning as
    _class_on_testid, which this file already grew for exactly this failure.
    """
    return [_class_value(tag, RAIL_TESTID) for tag in _rail_openings(provider)]


def _rail_openings(provider: str) -> list[str]:
    """The opening tag of each rail, comments already blanked."""
    clean = _without_comments(provider)
    tags = []
    at = clean.find(f'data-testid="{RAIL_TESTID}"')
    while at != -1:
        start, end = _opening_tag(clean, at)
        tags.append(clean[start:end])
        at = clean.find(f'data-testid="{RAIL_TESTID}"', end)
    return tags


# Anything that sets padding, in either of Tailwind's two spellings. The utility family
# (p-, px-, ps-, ...) and the arbitrary-property form, which Tailwind 4 emits with
# !important and which starts with "[" once _split_variants has taken the marker off, so a
# pattern anchored on "p" never sees it. The side letter is optional and a "-" must follow
# it either way, which is what keeps pointer-events-none, peer-* and place-items-* out.
_PADS = re.compile(r"p[xytblrse]?-|\[padding[-:]")


def _rail_class_tokens(tag: str) -> list[str]:
    """Every class token the rail's className CAN render, conditionals included.

    _class_value answers what renders in EVERY state, which is what a positive guarantee
    needs and exactly wrong for a prohibition: `cn("...", compact && "!pl-0")` renders that
    override whenever compact is true, and an always-rendered reader never sees it. So this
    one reads the literals out of the whole expression on purpose. The two are not
    interchangeable, and the asymmetry is the point: "must always have X" and "must never
    have Y" cannot be answered by the same set.
    """
    key = "className="
    at = tag.index(key) + len(key)
    if tag[at] == '"':
        expression = tag[at : _skip_literal(tag, at)]
    else:
        expression = tag[at + 1 : _balanced(tag, at) - 1]

    tokens: list[str] = []
    i = 0
    while i < len(expression):
        if expression[i] in _QUOTES:
            end = _skip_literal(expression, i)
            tokens += expression[i + 1 : end - 1].split()
            i = end
        else:
            i += 1
    return tokens


def _rail_padding(tag: str) -> dict[str, str]:
    """This rail's padding properties mapped to the constant each one is set from.

    The property-to-constant binding, not merely the presence of some STACK_ name: setting
    paddingLeft from STACK_CARD_INSET_RIGHT renders a 16px left gutter while the 22px floor
    below still reads a 28px constant nothing applies, and the shadow clips anyway.
    """
    return {
        prop: const
        for prop, const in re.findall(r"\b(padding(?:Top|Bottom|Left|Right)): (\w+)", tag)
    }


def _rail_style_px(provider: str, name: str) -> int:
    """The px value of one of the rail's spacing constants, read from its definition."""
    found = re.search(rf"^const {name} = (\d+);", provider, re.MULTILINE)
    assert found, f"{name} is gone from provider.tsx, so the rail's spacing is unreadable"
    return int(found.group(1))


def _capped_rails(provider: str) -> int:
    """How many of those rails cap themselves to the viewport.

    The cap is the full viewport and the gutters are paid out of it, as inline px padding,
    so the cards keep the band they had. Reading the class alone stopped being enough when
    #11260 moved the gutters out of the class and into the style, so this reads both.
    """
    # _only_under and not a substring: `md:max-h-[100dvh]` contains the utility while leaving
    # every smaller viewport uncapped, which is the spill this test exists to prevent.
    return sum(1 for rail in _corner_rails(provider) if _only_under(rail, "max-h-[100dvh]"))


def test_the_class_matchers_tell_a_gated_rule_from_an_ungated_one():
    """The floor assertions are only as strong as these, so they are tested.

    A matcher that quietly says yes is how this file went wrong the first time:
    the checks read as layout guarantees and were substring searches.
    """
    gated = "max-[383px]:has-[[data-slot=update-release-notes]]:min-h-[calc(1px+2px)]"
    assert _split_variants(gated) == (
        ("max-[383px]", "has-[[data-slot=update-release-notes]]"),
        "min-h-[calc(1px+2px)]",
    ), "a bracketed variant's own colon splits the token"
    # A variant that is asked for must be there, and the rest may be in any order.
    assert _applies(gated, "min-h-[calc(1px+2px)]", "max-[383px]")
    assert _applies(gated, "min-h-[calc(1px+2px)]", "max-[383px]", "has-[[data-slot=x]]") is False
    assert _applies("min-h-[calc(1px+2px)]", "min-h-[calc(1px+2px)]", "max-[383px]") is False
    # A utility is the whole last segment, not a suffix of one.
    assert _applies("min-h-0", "h-0") is False
    # And a gate cannot answer for a rule that has to hold everywhere.
    assert _applies("md:shrink-0", "shrink-0"), "the absence check must see a gated rule"
    assert _only_under("md:shrink-0", "shrink-0") is False
    assert _only_under("flex shrink-0 flex-col", "shrink-0")
    # A positive guarantee takes the gates it names and no others, in either
    # direction: one more narrows where the rule holds, and an ungated twin
    # widens it back over the state the gate exists to exclude.
    assert _only_under("has-[x]:min-h-4", "min-h-4", "has-[x]")
    assert _only_under("md:has-[x]:min-h-4", "min-h-4", "has-[x]") is False
    assert _only_under("has-[x]:min-h-4 min-h-4", "min-h-4", "has-[x]") is False
    assert _only_under("flex", "min-h-4", "has-[x]") is False, "absent is not satisfied"
    # An important rule is the same rule, at a weight that beats the floor, so
    # it cannot slip past a prohibition on the plain one.
    for important in ("!min-h-0", "min-h-0!"):
        assert _split_variants(important)[1] == "min-h-0"
        assert _applies(important, "min-h-0"), f"{important} escapes the prohibition"


def test_the_class_anchors_do_not_depend_on_any_order():
    """An anchor that needs an attribute or a branch to keep its place is the
    same brittleness one level up, so both are read structurally."""
    # An arrow function's `>` does not end the opening tag, and the attribute
    # is found on either side of the one that names the element.
    # A comparison inside an attribute expression is not the element's start.
    for tag in (
        '<ul className="a b" data-testid="x" onClick={() => go()}>',
        '<ul onClick={() => go()} data-testid="x" className="a b">',
        '<ul disabled={count < limit} className="a b" data-testid="x">',
        '<ul disabled={count < limit} data-testid="x" className="a b">',
    ):
        assert _class_on_testid(tag, "x") == "a b", tag
    # The two arms of the ternary are two different elements. A variant in the
    # first arm does not read as the separator, and only the second is returned.
    branch = _unpositioned_branch('positioned ? "fixed dark:bg-card" : cn("rail shrink-0")')
    assert "rail" in branch and "fixed" not in branch
    # A class expression renders the same DOM as a literal and must read the
    # same, but only what it renders in every state. A rule the card must
    # always carry is not satisfied by one that renders sometimes, and a
    # constant guard renders it never while leaving the text in the file.
    assert _class_value('<div className="a b">', "x") == "a b"
    assert _class_value('<div className={cn("a b")}>', "x") == "a b"
    assert _class_value('<div className={cn("a b", open && "c")}>', "x").split() == ["a", "b"]
    assert _class_value('<div className={cn(false && "a", "b")}>', "x").split() == ["b"]
    assert _class_value('<div className={cn(open ? "a" : "z", "b")}>', "x").split() == ["b"]
    # Comments are neither code nor classes. Prose can hold an apostrophe or an
    # unmatched brace, and the banners' own comment names `shrink-0`.
    for comment in ("// notes don't shrink", "/* an unmatched } is prose */"):
        tag = f'<ul {comment}\n className="a b" data-testid="x">'
        assert _class_on_testid(tag, "x") == "a b", comment
    blanked = _without_comments("/* shrink-0 */ flex")
    assert blanked.split() == ["flex"], "a class named in prose still reads as a class"
    assert len(blanked) == len("/* shrink-0 */ flex"), "blanking a comment moved every later index"
    # A `//` inside a literal is a URL, not a comment, so the rest of the line
    # and its closing quote survive.
    url = '"bg-[url(https://example.com/a.svg)] flex" // gone'
    assert _without_comments(url).rstrip() == '"bg-[url(https://example.com/a.svg)] flex"'


def test_the_overlay_stack_fits_the_viewport():
    """The card's own cap does not account for a download list stacked beneath
    it, so the rail carries one of its own. A static cap, not a measured one:
    a rail whose height and offset are computed from whatever else is on screen
    is a rail that moves out of its corner (#8082 and the chain after it)."""
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    # Counted by the layer they sit on, not by a literal z-index: the
    # overlay rail reads its depth from Z_LAYER now.
    stacks = provider.count("zIndex: Z_LAYER.OVERLAY_STACK")
    assert stacks, "the bottom-right overlay stack is gone"
    assert len(_corner_rails(provider)) == stacks, "a rail left its bottom-right corner"
    # Counted, not merely present: capping only one of the stacks is the bug here.
    assert _capped_rails(provider) == stacks, "every stack is capped"
    panel = (FRONTEND / "features/hub/download-manager/download-manager-panel.tsx").read_text(
        encoding = "utf-8"
    )
    # The download list scrolls internally, so it can give up height.
    assert "flex min-h-0" in panel
    # The update card cannot: its header and buttons are fixed and only its
    # notes yield, so it floors instead. The browser card floors itself off its
    # own content; the desktop card still states the floor as a constant.
    _assert_floors_itself(WEB_BANNER.read_text(encoding = "utf-8"), "browser")
    # Those floors can add up to more than the cap at a large type size, so the
    # rail scrolls. Without this the overflow lands below the bottom of the
    # screen with no way to reach it.
    assert provider.count("overflow-y-auto") >= stacks, "a capped stack spills its cards"


def test_both_rails_are_still_pinned_to_the_bottom_right_corner():
    """The corner, asserted rather than assumed by the matcher.

    _corner_rails finds rails by testid now, so it would happily return a rail that had
    wandered to the top left. This is the claim the old regex used to make implicitly, kept
    explicit and kept failing for the right reason: it names the rail that moved.
    """
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    rails = _corner_rails(provider)
    assert len(rails) == 2, f"expected the browser and desktop rails, found {len(rails)}"
    for rail in rails:
        # _only_under and not _applies: a positive layout guarantee has to hold everywhere, and
        # _applies is satisfied by a gated `md:fixed`, under whose breakpoint the rail would not
        # be in the corner at all. This file's own matcher tests spell that rule out.
        # The old class-anchored matcher required this as part of its pattern, and finding the
        # rail by testid instead dropped it silently. It is a behaviour contract, not styling:
        # the rail spans its cap with 28px of transparent shadow gutter and a scroll region,
        # and the cards inside opt back in with pointer-events-auto (the download panel does
        # so by name). Without the container rule those transparent bands swallow clicks meant
        # for the UI behind them.
        assert _only_under(
            rail, "pointer-events-none"
        ), f"the rail stopped passing clicks through: {rail!r}"
        assert _only_under(rail, "fixed"), f"the rail is no longer always viewport-fixed: {rail!r}"
        assert _only_under(rail, "bottom-0"), f"the rail can leave the bottom edge: {rail!r}"
        assert _only_under(rail, "right-0"), f"the rail can leave the right edge: {rail!r}"


def test_the_rail_gutters_come_out_of_the_cap_and_not_the_cards():
    """#11260's actual claim, which no class can carry any more.

    The rail caps at the whole viewport and pays its shadow gutters as inline px padding, so
    the band left for the cards is 100dvh less the two block gutters - the same band they had
    when the cap was written as calc(100dvh - 8px) and the gutter was 4px a side. px and not a
    spacing utility because those are rem and would scale the rail off its corner with the
    user's type size, which is the bug the comment above them is about.
    """
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    top = _rail_style_px(provider, "STACK_SHADOW_GUTTER_TOP")
    bottom = _rail_style_px(provider, "STACK_SHADOW_GUTTER_BOTTOM")
    left = _rail_style_px(provider, "STACK_SHADOW_GUTTER_LEFT")
    inset = _rail_style_px(provider, "STACK_CARD_INSET_RIGHT")

    assert top and bottom and left and inset, "a rail gutter went to zero, so shadows clip"
    # The deepest card shadow is 0 8px 28px -6px: 22px left of the card and 14px above it.
    assert left >= 22, f"the left gutter {left}px is inside the card shadow's 22px reach"
    assert top >= 14, f"the top gutter {top}px is inside the card shadow's 14px reach"
    expected = {
        "paddingTop": "STACK_SHADOW_GUTTER_TOP",
        "paddingBottom": "STACK_SHADOW_GUTTER_BOTTOM",
        "paddingLeft": "STACK_SHADOW_GUTTER_LEFT",
        "paddingRight": "STACK_CARD_INSET_RIGHT",
    }
    openings = _rail_openings(provider)
    assert len(openings) == 2, f"expected the browser and desktop rails, found {len(openings)}"
    for tag in openings:
        # The exact binding, per rail. Counting STACK_ names would let paddingLeft be set from
        # STACK_CARD_INSET_RIGHT: two constants, two rails, count still 2, and the floors above
        # would go on vouching for a 28px value nothing applies while 16px clips the shadow.
        assert (
            _rail_padding(tag) == expected
        ), f"a rail's padding is not bound to its own constant: {_rail_padding(tag)}"
    # No padding utility at all, in any spelling. The rail's padding comes from the inline
    # px style above, and ANY Tailwind padding class is either rem-scaled (walking the rail
    # off its corner with the user's type size, #8082) or, with !important, an outright
    # override of the inline declaration. Enumerating the spellings is how this went wrong
    # three times: the first form missed `!px-3`, the second `px-2.5` and `!pr-[0px]`, the
    # third the logical `ps-`/`pe-` pair this repo also uses. Matching the property rather
    # than its value ends that; `pointer-events-none` and `peer-*` do not match, since the
    # side letter is optional and a `-` has to follow it either way.
    for tag in _rail_openings(provider):
        for token in _rail_class_tokens(tag):
            utility = _split_variants(token)[1]
            assert not _PADS.match(utility), (
                f"the rail carries the padding utility {token!r}; its padding is the inline "
                f"px style, and a class here is rem-scaled or !important-overrides it (#8082)"
            )


def test_the_desktop_stack_is_capped_like_the_browser_one():
    """The download panel shares the desktop stack, left uncapped before now."""
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    assert len(_corner_rails(provider)) == 2, "both rails sit in the bottom-right corner"
    assert _capped_rails(provider) == 2, "both stacks are capped"
    tauri = TAURI_BANNER.read_text(encoding = "utf-8")
    _assert_floored(tauri, _SCALED_FLOOR_TAURI, _NARROW_FLOOR_TAURI, "desktop")


def test_the_rail_offset_is_not_computed():
    """The rail used to place itself around the boxes in the frame store, so a
    composer growing by a line or a download row arriving moved it to the middle
    of the window, and a maximised monitor to the top. Its offset and cap must
    stay out of JS."""
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    for banned in ("useStackGeometry", "stackGeometry", "stack.bottom", "stack.maxHeight"):
        assert banned not in provider, f"the rail is placed from JS again ({banned})"
    store = (FRONTEND / "features/settings/stores/monitor-frame-store.ts").read_text(
        encoding = "utf-8"
    )
    assert "stackBottomInset" not in store, "the dodge arithmetic is back in the frame store"


def test_desktop_notes_are_not_keyed_by_the_pinned_backend_version():
    """The banner asks with the Unsloth version it offers. `pypi_version` stays
    in latest.json as the backend pin preflight checks, not a notes key."""
    banner = TAURI_BANNER.read_text(encoding = "utf-8")
    assert "info?.version?.replace(LEADING_V" in banner
    assert "pypiVersion" not in banner, "notes are no longer keyed by the backend release"
    workflow = (REPO / ".github/workflows/release-desktop.yml").read_text(encoding = "utf-8")
    assert "'pypi_version': os.environ['PYPI_VERSION']" in workflow
    rust = (REPO / "studio/src-tauri/src/desktop_update_policy.rs").read_text(encoding = "utf-8")
    assert "pypi_version: Option<String>" in rust
    hook = NOTES_HOOK.parent.joinpath("use-tauri-update.ts").read_text(encoding = "utf-8")
    assert "rawPypiVersion(update.rawJson)" in hook


def test_link_resolver_reads_html_containers_the_way_the_others_do():
    """A `<details>` or `<div>` with no blank line inside is a type 6 block, so
    its contents are literal and a fence in it is not a fence, which stopped
    every link below from resolving. The parser and the preview already apply
    the type 6 and 7 rules, so the resolver has to share them."""
    links = LINKS.read_text(encoding = "utf-8")
    for source in (PREVIEW, LINKS):
        text = source.read_text(encoding = "utf-8")
        assert "HTML_BLOCK_TAGS" in text and "HTML_TAG_ONLY_LINE" in text
    # A blank line ends the block, not the closing tag, and a bare quote marker counts as blank.
    assert "inHtmlBlock = !!container.trim()" in links
    # Type 7 cannot interrupt a paragraph, so prose above it keeps its links.
    assert "return !afterParagraph && HTML_TAG_ONLY_LINE.test(line);" in links


def test_an_escaped_mark_makes_an_image_a_link():
    """`\\![alt](path)` renders as a link, so it resolves to the blob host."""
    links = LINKS.read_text(encoding = "utf-8")
    assert 'const image = bang === "!" && !isEscaped(line, offset);' in links
    # The reference pre-scan has to skip it too, or the definition flips host.
    assert "isEscaped(line, match.index)" in links


def test_link_resolver_reads_comments_before_fences():
    """A fence delimiter inside an HTML comment is not a fence: reading it as one
    left the fence open, so every link below went unresolved. The order matters
    both ways, so a comment opener inside a real fence is not a comment."""
    links = LINKS.read_text(encoding="utf-8")
    # Fence state is read before comments are masked, the order the collapsed preview uses.
    assert "const fenceSource = inComment\n      ? null\n      : FENCE.exec(" in links
    # Masking happens only after the in-fence early return.
    fence_return = links.index("// Fenced content is literal")
    assert links.index("const [line, stillInComment, stillRunOn] = maskComments(") > fence_return
    # Commented ranges join the code spans, so a hidden link is left alone.
    assert "const spans = [...codeSpans(masked), ...comments].sort(" in links


def test_preview_collects_labels_only_from_real_definitions():
    """A definition-shaped line inside indented code or a deep fence is literal,
    so recording its label made toPlainText strip brackets the expanded view
    keeps. The pre-scan skips the same code the collector pass skips; a real
    definition takes at most three spaces, so the indent test cannot reject one."""
    src = PREVIEW.read_text(encoding="utf-8")
    scan = src.index("const labels = new Set<string>();")
    collect = src.index("let deepFence: string | null = null;")
    prescan = " ".join(src[scan:collect].split())
    assert "let labelFence: string | null = null;" in prescan
    assert "if (line.indent - line.column >= INDENTED_CODE_INDENT) { continue; }" in prescan
    assert "endsDeepFence(labelFence, labelColumn, line)" in prescan


def test_the_download_panel_can_shrink_inside_the_capped_stack():
    """The stack is capped to the viewport and a flex item defaults to
    min-height:auto, so this wrapper could not shrink and the cap was absorbed by
    the fixed update card rather than the scrolling download list. Only the
    shared-stack branch needs it; standalone is fixed and not a flex item."""
    panel = (FRONTEND / "features/hub/download-manager/download-manager-panel.tsx").read_text(
        encoding="utf-8"
    )
    assert 'positioned ? "fixed bottom-4 right-4 z-50" : "flex min-h-0 justify-end"' in panel
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding="utf-8")
    # Counted by the layer they sit on, not by a literal z-index: the
    # overlay rail reads its depth from Z_LAYER now.
    stacks = provider.count("zIndex: Z_LAYER.OVERLAY_STACK")
    assert _capped_rails(provider) == stacks, "the cap this has to absorb"


@pytest.fixture(scope="module")
def run_scanner(tmp_path_factory):
    """Run the frontend's markdown scanners under node.

    Their job is to classify a line the way a CommonMark renderer would, which
    only a real run can show. The "@/lib" alias resolves through Vite, not node,
    so the copies have it rewritten."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is needed to run the TypeScript scanners")
    work = tmp_path_factory.mktemp("release-notes-scanners")
    for source in (PREVIEW, CODE_SPANS, LINKS, LIST_COLUMNS, INLINE_COMMENTS):
        rewritten = _TS_ALIAS.sub(r'"./\1.ts"', source.read_text(encoding="utf-8"))
        (work / source.name).write_text(rewritten, encoding="utf-8")
    (work / "run.ts").write_text(_TS_RUNNER, encoding="utf-8")

    def run(kind: str, markdown: str):
        result = subprocess.run(
            [node, "--experimental-strip-types", "--no-warnings", str(work / "run.ts"), kind],
            input=markdown,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"node could not run the scanners: {result.stderr.strip()[:200]}")
        return json.loads(result.stdout)

    return run


def preview_leads(preview) -> list[str]:
    return [item["lead"] for item in preview["items"]]


def test_a_link_indented_under_a_bullet_still_resolves(run_scanner):
    """CommonMark measures indentation from the container (spec 0.31.2 section
    5.2), so under "- Details:" a four-space line is two columns in: a paragraph
    holding a link. Measuring from the margin called it code (section 4.4) and
    left the destination relative to Unsloth's own origin."""
    resolved = run_scanner("links", "- Details:\n\n    [guide](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in resolved
    # The same prose one column further in really is code, and stays untouched.
    code = run_scanner("links", "- Added.\n\n      [guide](docs/a.md)\n")
    assert "[guide](docs/a.md)" in code and "github.com" not in code
    # At document level four spaces is code, so that link is still left alone.
    top = run_scanner("links", "Intro.\n\n    [guide](docs/a.md)\n")
    assert "[guide](docs/a.md)" in top and "github.com" not in top


def test_an_indented_fence_does_not_swallow_the_bullets_below_it(run_scanner):
    """A four-space line at document level is indented code, and a top-level
    bullet does not continue it. Promoting it to a list-contained fence left a
    block open with no closer, so every bullet after it was skipped."""
    swallowed = "Example:\n\n    ```\n\n- Added the exporter\n- Fixed the crash\n"
    assert preview_leads(run_scanner("preview", swallowed)) == [
        "Added the exporter",
        "Fixed the crash",
    ]
    # With nothing else to fall back on the summary disappeared entirely.
    assert preview_leads(run_scanner("preview", "    ```\n\n- Added the exporter\n")) == [
        "Added the exporter"
    ]
    # A fence that really is inside an item still hides that item's code.
    nested = "- a\n  - b\n    ```\n    - not a bullet\n    ```\n\n- Added tests\n"
    assert preview_leads(run_scanner("preview", nested)) == ["a", "Added tests"]


def test_a_table_only_release_previews_as_nothing(run_scanner):
    """A GFM table renders as a grid, so it previews as nothing and the panel
    stays collapsed. Falling through to the prose collector put the raw
    "| Change | Detail | | --- | --- |" delimiters in the popup instead."""
    table = "| Change | Detail |\n| --- | --- |\n| Exporter | Added GGUF |\n"
    assert run_scanner("preview", table)["items"] == []
    # A table after prose is dropped too, rather than joined onto it.
    assert preview_leads(run_scanner("preview", f"Some prose.\n\n{table}")) == ["Some prose."]
    # A bullet right after the rows ends the table, so it still previews.
    assert preview_leads(run_scanner("preview", f"{table}- Added tests\n")) == ["Added tests"]
    # Mismatched header and delimiter widths are no table, as on GitHub, so both lines are prose.
    assert preview_leads(run_scanner("preview", "| a | b |\n| --- |\n")) == ["| a | b | | --- |"]


def test_a_failed_fetch_keeps_retry_reachable():
    """A release with no notes is ready and a failed fetch is error; only the
    error is retryable, and rendering there would replace the Retry button."""
    src = " ".join(PANEL.read_text(encoding="utf-8").split())
    assert "const source = notes?.matched ? notes.markdown : null;" in src
    # Only NotesStatus renders retry, in the else of the markdown branch: an error has no markdown.
    assert "{markdown ? (" in src
    assert "retry={retry}" in src

    hook = " ".join(
        (FRONTEND / "hooks" / "use-release-notes.ts").read_text(encoding="utf-8").split()
    )
    assert (
        "const failed = !next || (!next.matched && next.error !== null);" in hook
    ), "the distinction this relies on"


def test_an_unclosed_comment_in_prose_cannot_hide_later_links(run_scanner):
    """CommonMark opens an HTML block (spec 0.31.2 section 4.6, type 2) only when
    the line begins with `<!--`; one mid-sentence is inline and cannot outlive
    its block. Carrying the unclosed state on masked every link below it."""
    repo = "https://github.com/unslothai/unsloth/blob/main/docs/a.md"
    # A separate list item is a separate block, so the link below still renders.
    item = run_scanner("links", "- Type <!-- to begin a comment\n- See [docs](docs/a.md)\n")
    assert repo in item
    # So does a paragraph the blank line already ended.
    paragraph = run_scanner("links", "Type <!-- to begin\n\nSee [docs](docs/a.md)\n")
    assert repo in paragraph
    # A delimiter inside inline code is literal, as it is for the parser.
    spanned = run_scanner("links", "- Wrap in `<!--` and `-->`\n- See [docs](docs/a.md)\n")
    assert repo in spanned
    # A comment starting a line is a block: it hides down to the closer's line, that line included.
    block = run_scanner("links", "<!-- staged\n- See [docs](docs/a.md)\n-->\n")
    assert repo not in block
    closer = run_scanner("links", "<!-- staged\n--> See [docs](docs/a.md)\n")
    assert repo not in closer


def test_a_parenthesised_link_destination_still_resolves(run_scanner):
    """A destination may hold parentheses while they balance (spec 0.31.2 section
    6.3), so `[x]((draft).md)` points at `(draft).md`. Stopping at the first
    paren matched an empty destination and left the link relative."""
    leading = run_scanner("links", "[details]((draft).md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/(draft).md" in leading
    # An image resolves against the raw host the same way.
    image = run_scanner("links", "![shield]((badge).png)\n")
    assert "https://raw.githubusercontent.com/unslothai/unsloth/main/(badge).png" in image
    # A pair in the middle of a path balances too.
    middle = run_scanner("links", "[api](docs/(v2)/api.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/(v2)/api.md" in middle
    # An unbalanced paren makes the destination invalid, so `[x](a(b.md)` is plain text, not a link.
    unbalanced = run_scanner("links", "[x](a(b.md)\n")
    assert unbalanced == "[x](a(b.md)\n"
    # One more closer balances the pair, and then it is a link again.
    closed = run_scanner("links", "[x](a(b.md))\n")
    assert "https://github.com/unslothai/unsloth/blob/main/a(b.md)" in closed
    # Pairs nest, and one level was all the expression allowed, so a path with two stayed relative.
    nested = run_scanner("links", "[x](((draft)).md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/((draft)).md" in nested
    deep = run_scanner("links", "![shot](((((v2))))).png)\n")
    assert "https://raw.githubusercontent.com/unslothai/unsloth/main/((((v2))))" in deep
    # The closer must still be there: an unbalanced run below a nested pair is not a link.
    across = run_scanner("links", "[x](((a).md\n[y](docs/y.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/y.md" in across
    assert "[x](((a).md" in across


def test_a_fence_inside_a_container_still_hides_its_sample(run_scanner):
    """A fence is measured from its container (spec 0.31.2 section 4.5), so
    `> ~~~` and one under a nested bullet both open a fence. Reading the margin
    never saw them, so a link in a code block was rewritten into verbatim text."""
    quoted = run_scanner("links", "> ~~~\n> [guide](docs/a.md)\n> ~~~\n")
    assert "[guide](docs/a.md)" in quoted and "github.com" not in quoted
    nested = run_scanner("links", "- a\n  - b\n    ~~~\n    [x](docs/x.md)\n    ~~~\n")
    assert "[x](docs/x.md)" in nested and "github.com" not in nested
    # A longer closer is still a closer, so the pair is not something a code span hid.
    uneven = run_scanner("links", "> ```\n> [guide](docs/a.md)\n> ````\n")
    assert "[guide](docs/a.md)" in uneven and "github.com" not in uneven
    # The fence ends with its container: a line outside the quote, or left of the item, is Markdown.
    left = run_scanner("links", "> ~~~\n[guide](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in left
    dedented = run_scanner("links", "- a\n  ~~~\n[guide](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in dedented
    # A document-level fence owns the quoted lines below, so the marker does not undo it.
    document = run_scanner("links", "~~~\n> [guide](docs/a.md)\n~~~\n")
    assert "[guide](docs/a.md)" in document and "github.com" not in document
    # Four columns past the item's content column it is indented code, not a fence: still literal.
    code = run_scanner("links", "- Details:\n\n      ~~~\n      [guide](docs/a.md)\n")
    assert "[guide](docs/a.md)" in code and "github.com" not in code


def test_an_html_block_inside_a_container_is_literal_too(run_scanner):
    """Type 1 and type 6 blocks are measured from their container too, so a
    `<details>` under a nested bullet and a `<pre>` in a quote are verbatim.
    Missing the opener rewrote the literal examples inside them."""
    nested = run_scanner("links", "- a\n  - b\n    <details>\n    [x](docs/x.md)\n    </details>\n")
    assert "[x](docs/x.md)" in nested and "github.com" not in nested
    quoted = run_scanner("links", "> <pre>\n> [x](docs/x.md)\n> </pre>\n")
    assert "[x](docs/x.md)" in quoted and "github.com" not in quoted
    # The block ends with its container, so a line dedented out of the item is Markdown again.
    dedented = run_scanner("links", "- a\n  - b\n    <details>\n[x](docs/x.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/x.md" in dedented
    # Inside a quote a bare marker holds nothing, the blank line that ends a type 6 block.
    blank = run_scanner("links", "> <details>\n>\n> [x](docs/x.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/x.md" in blank


def test_a_fence_written_as_an_item_first_content_opens_in_that_item(run_scanner):
    """A block straight after a marker is the item's own first content, measured
    from where that content starts (spec 0.31.2 section 5.2), so "- ```md" opens
    a fence. Reading the whole line never saw one, so the sample below was
    treated as prose and rewritten, and its info string became a headline."""
    sample = run_scanner("links", "- ```md\n  [example](docs/a.md)\n  ```\n")
    assert "[example](docs/a.md)" in sample and "github.com" not in sample
    ordered = run_scanner("links", "1. ~~~\n   [example](docs/a.md)\n   ~~~\n")
    assert "[example](docs/a.md)" in ordered and "github.com" not in ordered
    # The preview agrees: an item of only a code block previews as nothing; the next is a bullet.
    preview = run_scanner("preview", "- ```md\n  sample text\n  ```\n- Added tests\n")
    assert preview_leads(preview) == ["Added tests"]
    # One column further in it is indented code inside the item, so the link is prose and resolves.
    padded = run_scanner("links", "-     ```\n  [example](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in padded
    # A marker the paragraph above swallows opens no item, so no fence: ordered items open at 1.
    lazy = run_scanner("links", "Intro.\n2. ```\n[guide](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in lazy


def test_a_comment_may_close_on_a_later_line_of_its_paragraph(run_scanner):
    """A comment written mid-sentence belongs to its paragraph, so its `-->` may
    arrive on a later line of it and everything between renders as nothing.
    Ending it at its own line left a backtick inside pairing with a real one
    below, hiding a link, and left the preview quoting hidden text."""
    carried = run_scanner("links", "Note <!-- ` open\nstill --> see [d](docs/a.md) and `x`\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in carried
    # Text inside the comment renders as nothing, so it is left alone.
    inside = run_scanner("links", "Note <!-- see [c](docs/c.md)\nmore --> end\n")
    assert "[c](docs/c.md)" in inside and "github.com" not in inside
    # The preview hides it too, rather than quoting the comment at the reader.
    preview = run_scanner(
        "preview", "- Added X <!-- TODO: rewrite\n  this properly -->\n- Second\n"
    )
    assert preview_leads(preview) == ["Added X", "Second"]
    # An opener cannot outlive its paragraph: with it closed the `<!--` is text and hides nothing.
    broken = run_scanner("links", "Note <!-- open\n\nsecret --> end [d](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in broken
    # A heading breaks into the paragraph, so it ends the comment's reach too.
    headed = run_scanner("links", "Note <!-- open\n## 2.0 --> end [d](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in headed
    assert preview_leads(run_scanner("preview", "Note <!-- open\n\n- Second\n")) == ["Second"]


def test_only_punctuation_is_escapable_in_a_link_destination(run_scanner):
    """CommonMark escapes ASCII punctuation and nothing else (spec 0.31.2 section
    2.4), so the backslash in `docs\\alpha.md` is a character of the path.
    Dropping it rewrote a Windows or namespaced path to the wrong file."""
    kept = run_scanner("links", "[guide](docs\\alpha.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs%5Calpha.md" in kept
    # An escaped backslash is one literal backslash, which survives the same.
    escaped = run_scanner("links", "[guide](docs\\\\alpha.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs%5Calpha.md" in escaped
    # A real escape is still an escape: `\\(` is a paren of the path.
    paren = run_scanner("links", "[guide](a\\(b.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/a(b.md" in paren
    # A space still ends the destination, escaped or not, so there is no link.
    spaced = run_scanner("links", "[guide](a\\ b.md)\n")
    assert spaced == "[guide](a\\ b.md)\n"


def test_one_definition_does_not_hide_the_next(run_scanner):
    """Definitions may run consecutively (spec 0.31.2 section 4.7) and none is
    paragraph text, so none opens a paragraph the next cannot interrupt.
    Counting one as prose left every definition after the first unresolved.
    The backend already reads the line this way."""
    text = (
        "- AMD support is here, see [the AMD guide][amd] and the\n"
        "  [Intel notes][xpu].\n\n"
        "[amd]: docs/basics/amd.md\n"
        "[xpu]: docs/basics/xpu.md\n"
    )
    resolved = run_scanner("links", text)
    base = "https://github.com/unslothai/unsloth/blob/main/docs/basics/"
    assert f"[amd]: {base}amd.md" in resolved
    assert f"[xpu]: {base}xpu.md" in resolved
    # A run of them stays a run however long it is.
    run = run_scanner("links", "[a]: docs/a.md\n[b]: docs/b.md\n[c]: docs/c.md\n")
    assert run.count("https://github.com/unslothai/unsloth/blob/main/docs/") == 3
    # Prose between them opens a paragraph the next line may not interrupt, so it is not one.
    prose = run_scanner("links", "[a]: docs/a.md\nintro\n[b]: docs/b.md\n")
    assert "[b]: docs/b.md" in prose


def test_a_comment_closed_on_its_own_line_still_closes(run_scanner):
    """A multiline comment is usually closed by a `-->` on a line of its own, and
    a wrapped line may open with emphasis. Reading any leading punctuation as a
    new block meant neither continued the paragraph, so the comment never closed
    and the popup showed the author's internal note."""
    closer = run_scanner(
        "preview",
        "- DoRA training is available in Unsloth. <!-- TODO confirm the exact\n"
        "  flag name before release\n-->\n",
    )
    assert preview_leads(closer) == ["DoRA training is available in Unsloth."]
    # A continuation may open with emphasis, which is text and not a block.
    starred = run_scanner(
        "preview",
        "- DoRA training is available. <!-- TODO confirm the\n  *before* release -->\n",
    )
    assert preview_leads(starred) == ["DoRA training is available."]
    underscored = run_scanner(
        "preview",
        "- DoRA training is available. <!-- TODO confirm the\n  _draft_ note -->\n",
    )
    assert preview_leads(underscored) == ["DoRA training is available."]
    # A real block still ends the paragraph, so the opener below one is text and hides nothing.
    broken = run_scanner("links", "Note <!-- open\n## 2.0\nsecret --> [d](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in broken
    # So does a list item with content, which may interrupt a paragraph.
    item = run_scanner("links", "Note <!-- open\n- bullet\nsecret --> [d](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in item


# Real bodies, so the classification is checked against how releases are written.
