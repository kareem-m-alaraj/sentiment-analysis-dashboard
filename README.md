# Sentiment Analysis Dashboard

## Findings

Evaluation results so far. Ground truth is sentiment140's emoticon-derived
labels (0=negative, 4=positive) on twitter rows only; reddit has no
ground-truth labels.

### Zero-shot baseline

`cardiffnlp/twitter-roberta-base-sentiment-latest`, unmodified, is a 3-class
model (negative/neutral/positive) scored against sentiment140's 2-class
ground truth. Two different scoring populations, reported separately:

- **Full labeled set** (~50,000 twitter rows): 81.65% strict accuracy at
  73.62% coverage; 77.40% forced accuracy at 100% coverage.
- **Same 10,000-row held-out `test_split` used to evaluate the fine-tuned
  model below**: 81.27% strict accuracy at 73.93% coverage; 77.17% forced
  accuracy at 100% coverage.

Both populations are below an 85% target. The two are close (as expected —
`test_split` is a random subset of the full set), but only the second
figure is comparable to the fine-tuned model's score, below.

### Fine-tuned model

The 3-class head was replaced with a 2-class (negative/positive) head and
fine-tuned for 1 epoch on 20,000 sentiment140 rows (10,000/class). It scores
**86.76%** accuracy on the same 10,000-row held-out `test_split` (5,000/class)
used for the zero-shot comparison above.

The test post_ids were written to the `test_split` table before training
started and never regenerated; a direct check confirmed zero overlap
between the training set and `test_split`.

**86.76% (fine-tuned) vs. 81.27% (zero-shot strict) is the valid head-to-head
comparison** — both are scored on the identical 10,000 rows. The 81.65%
full-set figure above is not comparable to 86.76%; it's a different, larger
population that overlaps with the fine-tuned model's training data.

86.76% applies only to the 10,000-row held-out split — 20,000 of the
~50,000 twitter rows in the corpus were used as training data for this
model, so the same accuracy figure does not describe the full displayed
corpus.

### Zero-shot vs. fine-tuned agreement

Restricting to rows where the zero-shot model committed to negative or
positive (excluding its neutral calls), the two models agree on polarity:

- reddit: 86.60%
- twitter: 85.02%

This is an agreement rate between two models, not an accuracy figure —
reddit has no ground-truth labels, so there is no way to know which model
(if either) is correct on reddit rows.

### Calibration caveat: fine-tuned model on reddit

The fine-tuned model is a 2-class head and cannot output neutral. On reddit
text with no sentiment content — questions, logistics posts — it still
assigns a confident negative or positive label, sometimes above 0.9:

- "How do i invest in GameStop? I don't know anything about stock markets"
  → negative (0.979)
- "HOW CAN I BUY STOCK????!!!" → negative (0.928)

This follows directly from training data: sentiment140's labels are derived
from emoticons on a forced binary (negative/positive) scale, so the model
never saw a "no sentiment" class during training and has no way to express
one at inference time.

### Why the dashboard defaults to zero-shot

Despite the fine-tuned model's higher accuracy on the held-out test set,
the dashboard's default model is the zero-shot checkpoint. Reddit is the
majority of the displayed corpus and has no ground-truth labels; the
zero-shot model's ability to abstain (predict neutral) is preferred there
over the fine-tuned model's forced, sometimes falsely confident,
negative/positive call on sentiment-free text.
