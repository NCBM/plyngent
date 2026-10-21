# Deletes and the system trash

`delete_path` is the agent's only way to remove a path, and it is tagged
`YOLO`: under `--yes` / `/yolo on` it runs with no confirmation at all. So the
default has to be recoverable — a delete moves the path into the desktop's own
trash, where the file manager the human already uses can put it back.

```text
deleted file notes.md (moved to system trash)
deleted directory build (moved to system trash)
deleted file notes.md                               # permanent=true
```

`permanent=true` is the old behaviour, chosen on purpose. `[trash] enabled =
false` makes every delete that way.

## Where a path goes

| Platform | Trash |
|---|---|
| Linux / BSD | The freedesktop trash specification: `$XDG_DATA_HOME/Trash` (`~/.local/share/Trash` by default), or a per-volume trash when the path is on another filesystem |
| macOS | The Finder's trash (`FSMoveObjectToTrashSync`; `NSFileManager` when pyobjc is installed) |
| Windows | The Recycle Bin (`IFileOperation` with pywin32, else `SHFileOperation`) |

The platform work is [send2trash](https://github.com/arsenetar/send2trash), a
required dependency because a missing safety net that fails silently is worse
than no safety net. It is pure Python over `ctypes` — nothing to compile, no
runtime dependencies.

Nothing is renamed away piecemeal: the whole path moves into the trash in one
`rename`, so a large directory costs nothing to delete (across filesystems,
where the volume has no trash of its own, the fallback is a move into the home
trash, which does copy). A path trashed twice keeps both copies (`dup.txt`,
then `dup 1.txt`).

## What is deleted outright

| Case | Why |
|---|---|
| `permanent=true` | The call said so |
| `[trash] enabled = false` | The host said so |
| A path under the OS temporary directory | Scratch space: nobody restores it, and a trash on the `/tmp` volume would only collect it. The same rule that `new_temporary_workspace` cleans up with |
| `new_temporary_workspace` cleanup at chat exit | Same reason |
| `copy_path` / `move_path` with `overwrite=true` | The destination is being replaced by an explicit call; the source content is the point of the operation |

## When the trash is unavailable

A trashing that fails is reported, never downgraded to a deletion — the human
was told the path is recoverable, so guessing would be worse than stopping:

```text
error: cannot move to trash: Permission denied (use permanent=true)
error: cannot move to trash: send2trash is not installed (use permanent=true)
```

The usual causes are a path on a filesystem with no `.Trash` directory that the
user cannot create one on (network mounts, some FUSE filesystems, other
people's volumes), a parent directory without write permission, and a broken
install. In every case the path is untouched and the error names the way
forward.

## Restoring

Restore from the desktop's own trash — the file manager's "Trash" / "Recycle
Bin", which is where a human would look anyway. plyngent does not offer a
`restore` tool: `send2trash` does not report where a path landed, and the
platform trash formats differ. A delete therefore reports that the path is in
the trash, not where.

## What the trash does not cover

- **A shell command.** `run_argv(["rm", "-rf", …])`, a script, or anything run
  through a PTY reaches the filesystem directly; the command denylist and the
  soft confirm of interpreter runs are the only gates there, and `rm` does not
  use the trash.
- **Plugins and MCP servers.** They bring their own handlers.
- **Writes.** `write_file` with `overwrite=true` replaces a file's contents in
  place — it never removes a path. Only `delete_path`, the `copy_path` /
  `move_path` overwrite, and the temporary-workspace cleanup remove paths.

## Privacy

A trashed path is still on disk until the trash is emptied — that is the point
of it. `[agent].path_denylist` still runs first, so a denied path is never
trashed either, and a trashed file has left the workspace (it is no longer in a
`git status`, and no longer readable by a path tool).

## Platform limits worth knowing

- **Windows**: the shell deletes some things outright even when the call asked
  for the Recycle Bin — empty folders (a folder only reaches the bin with
  content), files above the bin's size limit for that drive, and paths on
  network shares. That is Windows' behaviour, not a fallback here.
- **Linux**: a path on another volume needs a trash directory there
  (`volume/.Trash/$UID` with the sticky bit, or a creatable `volume/.Trash-$UID`).
  Without one the file is moved into the home trash instead, which crosses a
  filesystem boundary and therefore copies.
- **macOS**: without pyobjc the library calls the older
  `FSMoveObjectToTrashSync`; installing `send2trash[nativeLib]` switches it to
  `NSFileManager`. Both trash; neither needs anything from plyngent.
