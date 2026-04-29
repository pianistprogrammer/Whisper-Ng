# Dataset Rebalancing & Distribution Strategy

## The Distribution Problem
During the curation of Common Voice datasets for Nigerian languages, initial evaluations revealed extreme statistical disparities and data-hoarding behaviors in the default static `.tsv` repositories (`train.tsv`/`test.tsv`).

* **Extreme Test Data Cannibalism:** Out of ~27,601 Hausa samples, over 10,755 of them (~39% of the dataset) were permanently siloed in `test.tsv`.
* **Resource Starvation:** Under-resourced languages like Yoruba and Igbo were similarly losing massive portions of usable domain adaptation parameters to testing buckets that failed to inform the model during active training.

## The Solution: Dynamic Concatenation and Proportional Re-Splitting
To resolve this loss of vital learning data, we designed a programmatic dataset restructuring algorithm that dynamically merges and redistributes the audio samples strictly based on a `70/15/15` ratio.

### Algorithmic Methodology:
We ignore rigid file-system splits and enforce dynamic allocation at runtime using `datasets.concatenate_datasets`:
1. **Unification:** `train.tsv` + `test.tsv` sets are concatenated into a monolithic `ds_local` entity.
2. **70:15:15 Math Split:** 
   * **70% Training:** We actively recover artificially locked testing data back into the `train` stream to feed the PEFT layer.
   * **15% Validation:** A mathematically defined partition dedicated to real-time Early Stopping.
   * **15% Testing:** A preserved held-out test block, symmetrical to Validation, used for post-process WER.

### Actual Output Data Allocations (70:15:15 Configuration)
By unlocking the `test` directories and applying the 70/15/15 ratio, Hausa training samples leaped from ~16,846 to ~19,300—an influx of ~2,500 highly valuable iterations!

| Language | Total Samples | Train (70%) | Validation (15%) | Test (15%) |
|----------|---------------|-------------|------------------|------------|
| **Hausa**| 27,601        | 19,320      | 4,140            | 4,141      |
| **Yoruba**| 2,574         | 1,801       | 386              | 387        |
| **Igbo** | 2,401         | 1,680       | 360              | 361        |
| **Pidgin**| 38            | 26          | 6                | 6          |

## Conclusion
Redistributing local low-resource speech datasets algorithmically forces symmetric evaluations and liberates critical domain data previously lost to misconfigured test repositories. The `hf/lora_whisper.py` script now handles this globally.
