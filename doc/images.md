# Images

A model that takes pictures can look at a screenshot, a chart, or a diagram
instead of reading a description of it. plyngent carries images as content
parts on messages; each wire surface lifts them into its own shape.

## Shape

`lmproto/openai_compatible/model.py`:

| Type | Meaning |
|---|---|
| `ChatImageUrl(url, detail)` | one picture: an `http(s)` link or an inline `data:` URL |
| `ChatImagePart(image_url)` | that payload as a `{"type": "image_url"}` block |
| `ChatTextPart(text)` | the text block beside images |
| `UserChatMessage.content` | `str`, or `list[ChatTextPart \| ChatImagePart]` when the turn carries pictures |
| `ToolChatMessage.images` | the pictures a tool returned, beside its text `content` |

Prose-only readers go through `content_to_text`, which renders each image as
`[image]` (summaries, transcripts, exports, the retry preview).

`agent/images.py` owns the bytes: it sniffs the format from the file's content
(JPEG `ff d8 ff`, PNG, GIF, WebP `RIFF….WEBP`), builds the data URL, and caps an
inline picture at `MAX_INLINE_IMAGE_BYTES` (8 MiB) — far below what a provider
accepts, because the message is re-sent every round and lives in the session's
JSON message column.

## Per surface

| Surface | A user turn's images | A tool result's images |
|---|---|---|
| chat completions (`compat.chat_messages_with_images`) | already content blocks | a `tool` message may only carry text, so the images move into the user turn emitted right after the tool run |
| Anthropic `/messages` (`messages_bridge`) | `image` blocks beside the text block | inside the `tool_result` block's content, beside a text block |
| Responses (`responses_bridge`) | `input_text` + `input_image` parts | in the `function_call_output` item's `output` array |

A chat-completions request drops the parts of a message it cannot express; no
history is rewritten (the stored message keeps its own images).

## Limits (provider side)

| Limit | Value |
|---|---|
| Formats | JPEG, PNG, GIF, WebP (sniffed from bytes, never the file name) |
| Request body | 48 MiB |
| One image, inline or URL | 32 MiB |
| One image, Files API `file_id` | 64 MiB |
| Images per request | 600 |
| Image side | 8192 px, or 4096 px when a request holds 15+ images |
| External URL | 8192 characters, downloaded within 60 s |
| Tokens per image | at most 1024 (a picture is resized to ~1300x1300 px, and one below ~544x544 is scaled up) |

Images belong to **user** messages only: a picture in a system or assistant
message is a 400 (`Image in assistant message is unsupported`). A tool result
lives inside a user turn, which is why it can carry one.

## Files API (upload once, reference by id)

Every surface we speak to can take a picture as an uploaded file instead of
inline bytes. The upload half of that is de-facto one shape; the object it
returns and the way a message refers to it are not standardized at all. Field
names below come from each vendor's own reference (DeepSeek `guides/files_api`,
OpenAI `resources/files` + `guides/file-inputs`, Anthropic `api/beta/files`).

| | OpenAI | Anthropic | DeepSeek |
|---|---|---|---|
| Upload | `POST /v1/files`, multipart: `file` + `purpose` (`user_data`, `vision`, …), optional `expires_after[anchor]` / `[seconds]` | `POST /v1/files`, multipart: `file` only; needs `anthropic-version` and `anthropic-beta: files-api-2025-04-14` | `POST /files`, OpenAI shape (`file` + `purpose="user_data"`, optional `expires_after[…]`); the same operations are also served under `/anthropic/v1/files` in the Anthropic shape |
| Id | `file-abc123` | `file_01HX…` | `file-api-…` |
| File object | `object:"file"`, `bytes`, `created_at` (unix seconds), `filename`, `purpose`, `expires_at` when it was asked for | `type:"file"`, `size_bytes`, `created_at` (RFC 3339), `filename`, `mime_type`, `downloadable` | both dialects: OpenAI shape on `/files`, Anthropic shape on `/anthropic/v1/files` |
| Reference in a message | Responses `{"type":"input_image","file_id":…}` (mutually exclusive with `image_url`); chat completions cannot reference an image by id at all — its `image_url` part has no `file_id`, and its `file` part is documents only | `{"type":"image","source":{"type":"file","file_id":…}}`, plus the beta header | chat: `{"type":"file","file_id":…}`, or inline `file_data` + `filename`; Anthropic: `source.type="file"` plus beta; Responses: `input_image.file_id` |
| Limits | 512 MB per file (what `purpose` allows) | not verified here (the API reference is unreachable from this host's region) | 64 MiB per file, 25 GiB and 10 000 files per account, expiry 1 h – 30 d or permanent |

What is shared: `POST …/files` as `multipart/form-data` with a `file` part,
and a string id back. What is not: the field names (`bytes` vs `size_bytes`),
the type tag (`object` vs `type`), the timestamp format, the cursor and paging
fields, the beta header, and every reference block above. A client therefore
needs one uploader and three reference shapes — the split this module already
has.

Three rules if this is ever wired in:

- a `file_id` is scoped to the provider (and account) that minted it, so a
  stored id belongs to the surface that produced it, exactly like
  `AssistantChatMessage.provider_blocks`; `/provider` cannot carry it across;
- OpenAI's chat-completions surface cannot reference an image id, so on that
  provider the feature implies the Responses convention;
- the token bill does not change: the model still processes the picture in
  every request that carries it. Uploading buys the 32 MiB inline cap, a
  smaller request body, and not re-sending the same screenshot each round.

## Verified against the DeepSeek API

The 2026-10-17 docs plus a live probe (`deepseek-flash`):

- the same picture is accepted on all three surfaces, and the usage is identical;
- an image inside `tool_result` content works, and the model reads it — the
  round that asked for a screenshot got the layout, the colours and the pixel
  span back;
- an image inside `function_call_output` works too (the round needs its
  `reasoning` item, which `responses_bridge` already sends);
- `detail: "low"` measured 3.4x cheaper than `original` on a 1200x900 picture
  (240 vs 708 prompt tokens) with no loss on "which colour where"; the floor is
  210 tokens even for a tiny picture, because small images are scaled up;
- re-sending a picture in a later round hits the provider's prefix cache
  (`cache_read_input_tokens`), so the "history is paid for again" cost is not
  the worst case;
- `/anthropic/messages` and `/anthropic/v1/messages` both answer, so the
  client's path stays as it is.

## Attaching one

Type `@path` in the line — no command, no separate step:

```
what is wrong with this layout? @shots/broken.png
```

`cli/mentions.py` reads the token (whitespace-delimited, `@"path with spaces"`
for names that hold one; an e-mail address is not a mention), resolves it against
the working directory, checks the path denylist, and attaches it. An image
becomes an image part; any other file is appended as text under a
`--- @path ---` header, capped like a tool result (its `truncate_token` keeps the
remainder resumable). A problem — a missing file, a denylisted path, a picture a
text-only model cannot take — prints a note and leaves the turn going.

Tab completes the path after `@` (directories end in `/`, a name with a space is
completed in its quoted form).

## Opting in

`[providers.<name>.models."<model>"].image_in` is the gate, and it covers every
way a picture can enter a turn — not only an `@image` mention:

| Path | Where it is gated |
|---|---|
| an `@image` mention (`cli/mentions.py`) | skipped, with a note naming the key |
| the `view_image` tool | `error: model '…' does not accept image input (set image_in = true under [providers.x.models."y"])` |
| an image body from `fetch_local` | the metadata stays, the picture is left behind with the same reason |
| an MCP server's image content, a plugin's `ToolOutput` | dropped by the registry's safety net (`agent/tools.py`), with the reason appended to the text |

`ReplState.sync_image_capability` publishes the selected model's verdict on
`InstanceState.workspace` and runs on every provider/model/client rebuild, so a
`/model` switch flips the gate at once. A host that binds no instance declares
nothing, and the picture is sent as usual.

The point is to fail loudly: without the gate a text-only model answers
`[Unsupported Image]` one round later, and a gateway that cannot read image
blocks hands the model raw base64 — which reads like the picture arrived when it
did not.

The default catalogs mark what takes pictures (`deepseek-flash` yes,
`deepseek-v4-pro` no, OpenAI's defaults yes). A model missing from the catalog
has to be declared: the flag is an explicit opt-in.

## Not implemented

- **Files API**: see above — no image path uses it today; a local 8 MiB cap plus
  resizing covers the common case, and the upload changes no token cost.
- **Image output**: the API documents none.
