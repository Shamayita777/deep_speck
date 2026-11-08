# CE5 amendment 01: OTS verification method (2026-11-08)

The amendment is `preregistration/amendment_01_ots_verification_method.json`. It replaces exactly one
production requirement: the recorded `ots verify` run, which needs a local Bitcoin node. The replacement
method is

**Java OpenTimestamps explorer-assisted verification plus independent second-endpoint raw-header corroboration.**

- **Not Bitcoin Core consensus verification.** No node validates the chain. Both block headers come from
  public explorers.
- **Not a two-endpoint check inside the Java CLI.** java-opentimestamps `8186391` (upstream HEAD on
  2026-11-08) queries the single endpoint `https://blockstream.info/api`. Its README describes a two-endpoint
  rule, but the code for it (`MultiInsight`) was removed in `91407bb` (2020-04-26). The second endpoint,
  `https://mempool.space/api`, is queried and checked by CE5 code.
- **Not the Python `ots verify`.**

The design, the approved preregistration (`0aa8fd61…`), its proof (`852fe259…`) and every other check are
unchanged. Production without `--amendment` still runs the registered gate, which refuses
(`gate_before_amendment.json`).

## Contents

| Path | What it is |
| --- | --- |
| `gate_before_amendment.json` | The registered gate's refusal at `44441f5`, recorded before any code changed |
| `phase_a_java_verify/` | Compatibility run: the official Java CLI on the exact approved JSON and proof. Exit code, stdout, stderr, runtime and CLI version, and file hashes before and after (unchanged) |
| `phase_a_mempool/` | Raw mempool.space replies for heights 974769 and 974770, and their local check (`mempool_corroboration.json`): double-SHA-256 of each 80-byte header, Merkle root and time compared with the proof and with the Java endpoint. The script that made it is kept as `.py.txt` so it does not enter CE5's registered source set |
| `phase_a_java_negative_wrong_file/` | The Java CLI on a one-byte-different file. It reports "File does not match original!" with a stack trace but exits 0, so exit code 0 alone is never accepted |
| `build/` | Build provenance: toolchain checksums, Adoptium metadata, Maven sha512, the source-archive hash, the build log, the built jars' hashes, and the hash of every jar in the isolated Maven repository |

The formal evidence record is
`preregistration/preregistration_409e7dcb_APPROVED.json.ots.java_explorer_corroboration.json`. It was written by
`runner javaots`, and `design.java_ots_verification_ok` judges it by recomputing everything from the record's
raw contents and from the proof.
