# Prompt notifications

A turn stops while a prompt waits on the human. If you switched to another
window, the wait is invisible until you come back — so the CLI announces it.

A prompt is announced only after `grace_seconds` of waiting (5s by default), so
answering at the terminal stays silent. Bursts are throttled by
`min_interval_seconds`.

A notification says what is asked, not what would be run. The title is a short
label; the body carries, in this order:

| Line | When |
|---|---|
| the question itself | an `ask_user_line` question, an `ask_user_choice` prompt |
| the reason line | only with `include_reason = true`: it can name a command, argv or URL, and a notification outlives the prompt (and may be mirrored to a lock screen) |
| the object | a confirmation or a batch: the program's name (`program: rm`), the file's (`target: todo.md`), the tools of a batch, `kind: directory` for an access request, the skill directory — never an argument, a URL or a piece of code |
| `no answer in 30s = deny` | a prompt that takes a default answer on timeout (command denylist, directory access, skill write) |

The lines are filled in that order until the body is full and each one is
capped, so a long reason drops out before the deadline does — the deadline is
the one line a human must not miss, because not answering *is* an answer there.
A desktop popup wraps and keeps the lines apart; the terminal channel carries
the same body on a single line.

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
| tool confirm (single) | Allow once / Deny / Go to terminal |
| a round's batch of confirms | Allow all / Review one by one / Deny / Go to terminal |
| command denylist, `fetch_local` private host | Allow once / Deny / Go to terminal |
| directory-access request | same three (the level asked for, not a lower one) |
| skill write | Allow once / This chat / Deny / Go to terminal |
| limit-continue | Continue / Continue, stop asking / Stop here / Go to terminal |
| `ask_user_line` | a reply field, plus Go to terminal — the answer *is* the text you type |
| `ask_user_choice`, `ask_user_form` | nothing (terminal only): the question and its options are in the body |
| PTY secrets | nothing at all: a secret prompt is never announced |

Each prompt says which answers it accepts (`prompt_wait(..., choices=…)`), and
those are the buttons; one that names none gets Allow once / Deny / Go to
terminal. "Go to terminal" is always the last button: it withdraws the
notification and leaves the prompt waiting — and so does a click on the
notification's body, which is the platform's own default action, so a
notification with no room for buttons still has that way back.

An answer read from the notification is treated exactly like a typed one —
`allow` is the `y` the prompt accepts, `deny` is `n` — and it never records a
session grant: the button means "this once", which is why a prompt with an
extra answer (this chat, stop asking) offers it as a button of its own. Text
typed into a reply field is an answer even when it spells a button's key
(`review` to a question is the word, not the button). A secret prompt is
announced nowhere and echoable from nothing: its value would have to travel
back through a surface that outlives it.

A prompt that can be answered from the notification reads both surfaces while it
blocks, from the first moment — the notification itself only shows up after
`grace_seconds`, so waiting for it to appear before listening would block on the
terminal alone. Typing at the terminal always wins; a readline-only convenience
(Tab completion on `ask_user_line`) is what a question gives up in exchange.

A prompt that *follows* an answer given on the notification — the "why denied"
comment, the per-call prompts after "Review one by one" — is announced at once,
with no grace period and past the burst throttle: the human is looking at the
notification, and the follow-up is the next thing they must answer.

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

The notification stays on screen: it is sent with no expiry (`timeout = 0`, the
freedesktop value for "until dismissed"), so the server does not slide it into
the tray a few seconds later. That is the point of it — the turn is blocked
until the prompt is answered, and a collapsed notification is one the human
would find only on coming back anyway. Urgency stays at the library default: on
KDE Plasma, `critical` keeps the notification up just the same but paints the
popup as urgent, and on every platform it asks the Focus mode to let it through
— which a prompt has no business doing. Once the prompt is answered, the
notification is dismissed.

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
