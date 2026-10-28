# Bonsai 2 27B on a GTX 1070

Launchers for this branch's build. Paths outside the repo (CUDA toolkit, models, web UI) come from `BONSAI2`
(default `C:\git\0\bonsai2`): `%BONSAI2%\cuda`, `%BONSAI2%\models`, `%BONSAI2%\web`.

| Script | Setup | Decode t/s (desktop on the card) |
|---|---|---|
| `build-v2.cmd` | builds this tree into `build-v2` (sm_61) | |
| `run-server-mtp-32k.cmd` | MTP n-max 3 + stop rule, 32k, q8_0 K / q4_0 V | ~36 chat, 24.1 at 27k tokens |
| `run-server-mtp-24k.cmd` | MTP n-max 2, 24k, q8_0 KV | 34.6 chat, 30.2 at 16.5k tokens |
| `run-server-48k.cmd` | no MTP, 48k, q8_0 KV | ~21 |

`tables-n3-v6.txt` holds the stop-rule tables used by the 32k launcher. Extra arguments are passed to llama-server.
