# TASK-0255 Testing

## Verified results

| Probe | Pooled P/R/F1 | Hazen P/R | VTV P/R | Randolph P/R |
| --- | --- | --- | --- | --- |
| MViT | 0.667 / 0.143 / 0.235 | 0 / 0 | 0.5 / 0.25 | 1 / 0.25 |
| Swin | 0.875 / 0.5 / 0.636 | 0.833 / 0.833 | 1 / 0.5 | 0 / 0 |
| MViT+Swin | 0.889 / 0.571 / 0.696 | 1 / 1 | 0.667 / 0.5 | 0 / 0 |

Read-only ranking diagnostics for the concatenated probe are pooled ROC-AUC
`0.883929` and average precision `0.937527`. Randolph itself retains ROC-AUC
`0.833333` and average precision `0.8875`, but its maximum held probability
`0.815851` is below the independently selected threshold `0.845793`. This is
evidence of cross-source score calibration/domain shift, not evidence that the
85% decision gate passed.

Both new probes passed the strict verifier and retained
`runtime_consumable=false`, `formal_evaluation_eligible=false`,
`promotion_eligible=false`, and `promoted=false`.
