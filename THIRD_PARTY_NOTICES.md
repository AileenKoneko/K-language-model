# Third-Party Notices

This repository contains original code by AileenKoneko licensed under the MIT license in [LICENSE](LICENSE), plus specific third-party derived material noted below.

## RWKV-v8 ROSA

Files:

- `k_language_model/rosa.py`
- `k_language_model/rosa_backends.py`
- `k_language_model/rosa_ext.cpp`

Attribution:

- These files include implementation material derived from the RWKV-v8 ROSA formulation and pseudocode published by BlinkDL and the RWKV project.
- The implementation in this repository changes the architectural injection point: here, `rosa_h` is computed once and injected into selected K2 mixing layers with learned `rho` gating, rather than being applied at the embedding/input side.

License treatment:

- The derived ROSA material is carried with Apache License 2.0 attribution.
- The Apache 2.0 license text is included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

Sources:

- RWKV-8 ROSA article image/text: <https://www.rwkv.com/images/RWKV-8-ROSA.png>
- RWKV-v8 ROSA directory: <https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8>
- RWKV-8 note: <https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-8.md>
- RWKV FAQ: <https://wiki.rwkv.com/basic/FAQ.html>
- RWKV blog post: <https://blog.rwkv.com/p/rwkv-joins-the-linux-foundation-as>
