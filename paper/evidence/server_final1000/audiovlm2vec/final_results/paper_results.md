# Audio-as-Text VLM2Vec Reference Diagnostic

This is an independent reproduction built from the public VLM2Vec-Qwen2VL-7B adapter and
Qwen2-Audio captions. It is not the unreleased official AudioVLM2Vec checkpoint.

- Audio-CVR Test1000 SHA256: `70bd998c33bd4c2168ac18afb26ec6fbe928b234c61241f53412be387d52ec9e`
- Audio-CVR queries: 1000
- OmniCVR queries: 1000

## audiocvr

| Model | Mode | Source | R@1 | R@5 | R@10 | Target>Source | Gap |
|---|---|---|---:|---:|---:|---:|---:|
| zero-shot | V_T | with | 0.1100 | 0.7450 | 0.8310 | 0.1620 | -0.0230 |
| zero-shot | V_T | masked | 0.5740 | 0.7640 | 0.8420 | 0.1620 | -0.0230 |
| zero-shot | V_A_T | with | 0.0220 | 0.1320 | 0.1620 | 0.2320 | -0.0441 |
| zero-shot | V_A_T | masked | 0.0930 | 0.1360 | 0.1650 | 0.2320 | -0.0441 |
| adapter mean±std | V_T | with | 0.0258±0.0032 | 0.0828±0.0084 | 0.1140±0.0101 | 0.4320±0.0160 | -0.0111±0.0026 |
| adapter mean±std | V_T | masked | 0.0476±0.0078 | 0.0866±0.0078 | 0.1176±0.0094 | 0.4320±0.0160 | -0.0111±0.0026 |
| adapter mean±std | V_A_T | with | 0.0224±0.0062 | 0.0458±0.0050 | 0.0624±0.0040 | 0.5060±0.0066 | 0.0504±0.0076 |
| adapter mean±std | V_A_T | masked | 0.0244±0.0051 | 0.0470±0.0057 | 0.0630±0.0046 | 0.5060±0.0066 | 0.0504±0.0076 |

## omnicvr

| Model | Mode | Source | R@1 | R@5 | R@10 | Target>Source | Gap |
|---|---|---|---:|---:|---:|---:|---:|
| zero-shot | V_T | with | 0.0110 | 0.3010 | 0.4320 | 0.0240 | -0.1265 |
| zero-shot | V_T | masked | 0.1370 | 0.3300 | 0.4540 | 0.0240 | -0.1265 |
| zero-shot | V_A_T | with | 0.0850 | 0.3360 | 0.4200 | 0.2910 | -0.0648 |
| zero-shot | V_A_T | masked | 0.1670 | 0.3480 | 0.4290 | 0.2910 | -0.0648 |
| adapter mean±std | V_T | with | 0.0176±0.0041 | 0.0682±0.0141 | 0.0968±0.0211 | 0.3020±0.0263 | -0.1003±0.0071 |
| adapter mean±std | V_T | masked | 0.0288±0.0062 | 0.0722±0.0163 | 0.0990±0.0226 | 0.3020±0.0263 | -0.1003±0.0071 |
| adapter mean±std | V_A_T | with | 0.0090±0.0019 | 0.0268±0.0056 | 0.0398±0.0087 | 0.4094±0.0137 | -0.0701±0.0094 |
| adapter mean±std | V_A_T | masked | 0.0094±0.0015 | 0.0270±0.0058 | 0.0398±0.0087 | 0.4094±0.0137 | -0.0701±0.0094 |

