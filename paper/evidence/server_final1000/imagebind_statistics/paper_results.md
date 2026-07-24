# ImageBind Audio-CVR Results

ImageBind-Huge is evaluated zero-shot with equal-weight normalized modality arithmetic.

| Mode | With-ref R@1 | Without-ref R@1 | Ref-induced drop | R@5 | R@10 | Target beats ref | Gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| T_only_fullAV | 0.0210 | 0.0320 | 0.0110 | 0.0960 | 0.1520 | 0.5220 | 0.00433 |
| V_only | 0.0090 | 0.9960 | 0.9870 | 0.9990 | 1.0000 | 0.0000 | -0.01323 |
| A_only | 0.0000 | 0.9420 | 0.9420 | 0.9730 | 0.9830 | 0.0000 | -0.07324 |
| V_T | 0.1170 | 0.9930 | 0.8760 | 0.9980 | 0.9980 | 0.1080 | -0.00767 |
| A_T | 0.0410 | 0.9290 | 0.8880 | 0.9760 | 0.9850 | 0.0410 | -0.04377 |
| V_A | 0.0000 | 0.9810 | 0.9810 | 0.9920 | 0.9960 | 0.0000 | -0.03267 |
| V_A_T | 0.0250 | 0.9850 | 0.9600 | 0.9940 | 0.9960 | 0.0250 | -0.02334 |

## Paired tests

- `audio_gain_R@1`: delta=-0.09200, 95% CI=[-0.113, -0.071], p=4.9998e-05, Holm p=0.00019999.
- `audio_gain_target_reference_gap`: delta=-0.01567, 95% CI=[-0.017653052900731565, -0.01383409005403519], p=4.9998e-05, Holm p=0.00019999.
- `V_A_T_reference_masking_R@1`: delta=0.96000, 95% CI=[0.947, 0.972], p=4.9998e-05, Holm p=0.00019999.
- `V_T_reference_masking_R@1`: delta=0.87600, 95% CI=[0.855, 0.896], p=4.9998e-05, Holm p=0.00019999.
