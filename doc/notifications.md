# Prompt notifications

A turn stops while a prompt waits on the human. If you switched to another
window, the wait is invisible until you come back — so the CLI announces it.

A prompt is announced only after `grace_seconds` of waiting (5s by default), so
answering at the terminal stays silent. Bursts are throttled by
`min_interval_seconds`, and the announcement says what is being asked, not what
it would run: the one-line reason can name a command, argv or URL, and a
notification outlives the prompt (and may be mirrored to a lock screen), so it
travels only with `include_reason = true`.

## Configuration

```toml
[notify]
enabled = true
grace_seconds = 5.0
min_interval_seconds = 10.0
channels = ["auto"]        # auto | desktop | terminal
actions = true             # offer the answer, not just the news
include_reason = false
protocol = ""              # terminal protocol override: osc777 / osc9 / bell
```

`channels = ["auto"]` tries the desktop first and falls back to the terminal;
`["desktop"]` and `["terminal"]` pin one of them. Notifications are only sent
from an interactive run — a one-shot `plyngent "…"` never waits on a human, and
a prompt answered inside the grace period is never announced at all.

## What is answered where

| Prompt | Notification offers |
|---|---|
| tool confirm (single / a round's batch) | Allow once / Deny / Go to terminal |
| command denylist, `fetch_local` private host | same three |
| directory-access request, skill write | same three |
| `ask_user_line` | a reply field — the answer *is* the text you type |
| `ask_user_choice`, `ask_user_form`, PTY secrets | nothing (terminal only) |
| limit-continue | nothing (terminal only) |

An answer read from the notification is treated exactly like a typed one —
`allow` is the `y` the prompt accepts, `deny` is `n` — and it never records a
session grant: the button means "this once". "Go to terminal" withdraws the
notification and leaves the prompt waiting. A secret is never asked for in a
notification, and never echoed into one.

A prompt that can be answered from the notification reads both surfaces while it
blocks, from the first moment — the notification itself only shows up after
`grace_seconds`, so waiting for it to appear before listening would block on the
terminal alone. Typing at the terminal always wins; a readline-only convenience
(Tab completion on `ask_user_line`) is what a question gives up in exchange.

## Desktop channel

The desktop channel needs the optional extra:

```console
pip install 'plyngent[notify]'
```

It wraps `desktop-notifier`, which talks to `org.freedesktop.Notifications` on
Linux, Notification Center on macOS and WinRT on Windows. Buttons need a
notification server that renders them (KDE Plasma and GNOME Shell do; a bare
`dunst` may not, and then the notification is text-only without any error). The
reply field is KDE-only in practice: it is the `inline-reply` capability plus the
`x-kde-reply-submit-button-text` hint, which only KNotifications implements.

Notifications are shown from the library's async API on a loop that runs in a
daemon thread, because callbacks only fire while the loop that owns the
notification is running.

## Terminal channel

No dependency and no extra: the CLI writes an escape sequence to `/dev/tty`
(never stdout, so a redirect cannot swallow it) and the terminal turns it into a
native notification.

| Protocol | Sequence | Terminals |
|---|---|---|
| OSC 777 | `ESC ] 777 ; notify ; title ; body` | Ghostty, WezTerm, Warp, Contour, VTE (GNOME Terminal, Tilix, Terminator), rxvt, Windows Terminal |
| OSC 9 | `ESC ] 9 ; title: body` | iTerm2, Apple Terminal, ConEmu |
| bell | `BEL` | everything else |

Detection reads the environment only (`TERM_PROGRAM`, `TERM`, `WT_SESSION`,
`VTE_VERSION`), so it never blocks and never touches the input stream;
`protocol` pins it. Inside tmux the sequence is wrapped in DCS passthrough —
tmux needs `set -g allow-passthrough on` — and inside GNU Screen in a Screen
DCS. kitty speaks OSC 99 only, which is not implemented, so it keeps the bell;
install the extra to reach it through the desktop instead.

## Failure is always silent

A missing extra, a desktop without a notification server, a refused write: each
falls through to the next channel and never raises into the prompt. There is no
sound, no remote push and no notification when a turn merely ends — only when a
prompt is waiting for you.
