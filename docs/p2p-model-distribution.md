# Peer-to-peer model distribution

When more than one node in a cluster needs the same model, exo can have
nodes that already hold the weights serve them to peers over the local
network instead of every node fetching independently from HuggingFace.
On a Thunderbolt-meshed cluster this means a 200 GB model is pulled from
the internet exactly once and then fans out at link-local speed.

## How it works

Each node runs a small HTTP file server on `EXO_FILE_SERVER_PORT`
(default `52416`) that serves files under any configured model directory
(`EXO_MODELS_DIRS` and `EXO_MODELS_READ_ONLY_DIRS`). When a node is
asked to download a model, the master and the worker both consult
`find_peer_repo_url` (`src/exo/download/peer_discovery.py`):

1. Look at the global `download_status` for any peer that is in
   `DownloadCompleted` for that exact model.
2. From that peer's `node_network` info, pick the best routable IPv4
   address, ranking interface types
   `thunderbolt > maybe_ethernet > ethernet > everything else`.
   IPv6, loopback (`127.0.0.0/8`) and IPv4 link-local
   (`169.254.0.0/16`) addresses are skipped.
3. Return `http://<peer-ip>:<port>` as the `repo_url` to use for the
   download.

If no peer has the model, `repo_url` is left `None` and the downloader
falls through to the existing HuggingFace flow.

The downloader (`_download_file_from_peer` in
`src/exo/download/download_utils.py`) shells out to `curl` with
`-C -` (auto-resume) and `-f` (fail-fast on HTTP error). Per-file
parallelism is bumped to 16 concurrent files when running against a
peer (vs the conservative HF default), since the bottleneck is local
NIC throughput rather than HF rate limits.

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `EXO_FILE_SERVER_PORT` | `52416` | TCP port the per-node file server listens on. |
| `EXO_FILE_SERVER_BIND_HOST` | `0.0.0.0` | Interface the file server binds to. Set to `127.0.0.1` to disable P2P serving on this node, or a specific interface IP to narrow exposure. |
| `EXO_FILE_SERVER_MAX_CONCURRENCY` | `64` | Cap on concurrent in-flight serves. Excess requests get 503 with `Retry-After: 1`. Sized for a 4-peer fan-out × 16 files-per-peer concurrent download (matches the receiver's per-shard parallelism). |
| `EXO_MODELS_DIRS` | `~/.cache/exo/models` | Colon-separated writable model dirs. The file server will look for each requested model in each of these. |
| `EXO_MODELS_READ_ONLY_DIRS` | _(empty)_ | Colon-separated read-only model dirs (e.g. NFS mirrors). Also served by the file server. |

## Runtime requirement

The peer-to-peer download path shells out to `curl`. It must be on
`$PATH` of every worker that wants to use P2P. macOS and most Linux
distros ship it preinstalled; on a minimal container image you'll
need `apt-get install curl` (or equivalent) before the worker starts.

If `curl` is missing, downloads will fall through to the HuggingFace
path (no P2P speedup), but the existing flow continues to work.

## Security stance

**The file server has no authentication.** Anything you put in
`EXO_MODELS_DIRS` or `EXO_MODELS_READ_ONLY_DIRS` is served to anyone
who can reach `EXO_FILE_SERVER_PORT` on the host. The server binds to
`0.0.0.0` by default because peer IPs are discovered dynamically from
each node's `NodeNetworkInfo`, so the binding can't be narrowed in
advance.

This is intentional for the assumed deployment — a private cluster on
trusted infrastructure (a dedicated Thunderbolt mesh, a private VPC)
where the operator already controls who can reach the host. **Do not
expose `EXO_FILE_SERVER_PORT` on a host reachable from the public
internet** without putting your own auth proxy in front of it.

If you need a tighter posture:

- Block the port at the host firewall (`pf` / `nftables`) and only
  allow the IP range your peer mesh actually uses.
- Run exo nodes inside a private VPC / VPN and never expose
  `EXO_FILE_SERVER_PORT` on a public interface.
- Set a non-default `EXO_FILE_SERVER_PORT` to dodge naive scanners,
  though this is obfuscation, not a control.

### What the server does and doesn't defend against

It does:

- Pin every served file to a *specific normalized model subdirectory*,
  not just to "somewhere under a model dir". A `..`-style traversal
  that escapes that subdirectory — even sideways into another model
  inside the same root — is rejected with 404, not by erroring out
  on the symptom but by checking `is_relative_to` against the exact
  expected subdir.
- Reject HTTP traversal sent at the byte level (literal `..`).
  We test this with a raw-socket request, not aiohttp's URL-normalizing
  test client, because clients silently collapse `..` and would
  otherwise hide a server-side bug.
- Quietly ignore malformed `Range` headers (`bytes=abc-`,
  `bytes=10-20,30-40`, suffix-form `bytes=-100`) instead of crashing
  the request handler. Pre-fix this was a trivial DoS — a single
  malformed header raised an uncaught `ValueError`.
- Refuse to echo request-path bytes back in error responses.
  `text/plain` makes XSS-by-itself unlikely, but reflecting attacker
  input is bad practice and we don't.

It does (continued):

- **Hash-verify P2P-downloaded bytes when the source has a sidecar.**
  After the HF download path completes its own SHA256 check, it
  writes a `<file>.sha256` sidecar next to the file. The file server
  reads the sidecar and emits its contents in an `X-File-SHA256`
  response header. The receiver captures the header (via
  `curl -D <file>` so it's a single round-trip), hashes the
  downloaded bytes, and refuses to rename `.partial` → final on
  mismatch. The outer retry loop re-invokes the download, and `-C -`
  auto-resumes — but on a hash mismatch we delete the partial first,
  so the retry effectively starts over (and may switch peers if a
  different one is available). When the source has no sidecar (older
  build, file fetched before sidecar-writing landed) the header is
  omitted and the receiver proceeds without verification, logged at
  debug. This catches transmission corruption and disk-side decay on
  the source. It does **not** defend against a *compromised* peer
  that recomputes the hash of substituted bytes.
- **Cap concurrent serves.** A misbehaving peer that spawns enough
  parallel `curl` fetches to saturate the host gets bounced with
  `503 Retry-After: 1` once `EXO_FILE_SERVER_MAX_CONCURRENCY` is
  reached. The receiver's existing retry loop handles the 503 by
  re-issuing curl, which auto-resumes — so over-cap requests don't
  lose any bytes, they just queue at the receiver instead of
  amplifying memory pressure on the server.

It doesn't:

- **Defend against a malicious peer.** Hash verification only
  protects integrity *given* an honest sidecar. A compromised peer
  can compute and serve a hash for whatever bytes it likes. The trust
  boundary remains "every node in your cluster is trusted." If that
  trust ever breaks, you have bigger problems than P2P weight
  distribution.
- **Rate-limit total bandwidth.** The concurrency cap limits the
  number of in-flight serves, not the bytes/sec each one can move.
  This is intentional — on a TB-mesh the entire point is to use the
  full link bandwidth. If you need bytes/sec throttling, do it at
  the OS (`tc`/`pf`) level.

## State machine note

This PR also separates the `DownloadPaused` state from
`DownloadPending`. Cancelling an active download was previously
modelled as a transition back to `DownloadPending`, the same state
used for "we haven't started yet"; this conflated _paused with bytes
on disk_ with _never ran_. The two are now distinct, and the
worker plan loop deliberately doesn't auto-restart `DownloadPaused`
the way it does `DownloadPending`.
