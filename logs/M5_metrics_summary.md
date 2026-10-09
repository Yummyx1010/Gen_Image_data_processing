# M5 metric summary

The original A, F and B models were evaluated on 8,666 test images.

| Model | glide AUROC | Midjourney AUROC | wukong AUROC | Mean AUROC | Cross-generator SD | Worst AUROC |
|---|---:|---:|---:|---:|---:|---:|
| A: spatial | 0.9746 | 0.9369 | 0.9429 | 0.9515 | 0.0166 | 0.9369 |
| F: frequency | 0.9166 | 0.7105 | 0.7953 | 0.8074 | 0.0851 | 0.7105 |
| B: fusion | 0.9758 | 0.9075 | 0.9400 | 0.9411 | 0.0280 | 0.9075 |

Values summarise seeds 42, 123 and 2026. Higher AUROC is better. Lower cross-generator SD means more similar performance across the three generators, and must be interpreted alongside AUROC.

## Results

B had lower mean test AUROC than A in all three seeds. Averaged across seeds, mean AUROC was 0.9411 for B and 0.9515 for A. Cross-generator SD was higher for B (0.0280) than A (0.0166), showing greater variation in AUROC across the three generators. B also had lower worst-generator AUROC (0.9075 versus 0.9369).

Averaged across seeds, B was slightly better on glide (0.9758 versus 0.9746), but lower on Midjourney (0.9075 versus 0.9369) and wukong (0.9400 versus 0.9429). The largest decrease was on Midjourney. The small gain on glide did not offset the declines on the other generators.

F had the lowest mean AUROC (0.8074) and the highest cross-generator SD (0.0851) of the three models.

## Validation and unseen-generator testing

Training uses BigGAN, stable_diffusion_v_1_5 and ADM; validation uses VQDM; testing uses glide, Midjourney and wukong. Each split also contains Nature real images.

B achieved a higher best-checkpoint validation AUROC than A in every seed, but lower mean test AUROC in every seed. The improvement on VQDM did not carry over to average performance on glide, Midjourney and wukong.

## Conclusion

Under the evaluated data, model and training settings, adding a frequency-domain branch did not improve average detection performance or cross-generator consistency on the three unseen generators. This shows that adding frequency information does not necessarily improve generalization to unseen generators.

The comparison evaluates the added branch as a whole, including its extra trainable parameters. No statistical significance test was performed.

## Definitions

Each generator evaluation uses 2,500 AI images and the same 1,166 real images.

For each seed, mean AUROC is the average of the three generator AUROCs, cross-generator SD is their population SD (ddof=0), and worst-generator AUROC is their minimum. The table averages these statistics across seeds.

Metric details and reproduction commands are in [SCORING.md](../m5/SCORING.md).
