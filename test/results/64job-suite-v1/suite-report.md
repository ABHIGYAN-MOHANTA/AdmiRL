# Tuned 5-Seed Kueue Suite

## Arm Means

### cohort + BestEffortFIFO

- `head_gang_blocked_seconds`: mean `467.400`
- `avg_gang_wait_seconds`: mean `467.400`
- `avg_gang_completion_seconds`: mean `475.400`
- `avg_elastic_wait_seconds`: mean `0.000`
- `avg_elastic_completion_seconds`: mean `0.000`
- `p95_gang_wait_seconds`: mean `467.400`
- `makespan_seconds`: mean `475.582`
- `throughput_jobs_per_minute`: mean `8.074`
- `small_job_bypass_count_while_gang_pending`: mean `62.000`
- `avg_small_wait_seconds`: mean `218.197`

### cohort + StrictFIFO

- `head_gang_blocked_seconds`: mean `385.200`
- `avg_gang_wait_seconds`: mean `385.200`
- `avg_gang_completion_seconds`: mean `393.200`
- `avg_elastic_wait_seconds`: mean `0.000`
- `avg_elastic_completion_seconds`: mean `0.000`
- `p95_gang_wait_seconds`: mean `385.200`
- `makespan_seconds`: mean `393.382`
- `throughput_jobs_per_minute`: mean `9.768`
- `small_job_bypass_count_while_gang_pending`: mean `62.000`
- `avg_small_wait_seconds`: mean `179.257`

### cohort + learned

- `head_gang_blocked_seconds`: mean `18.000`
- `avg_gang_wait_seconds`: mean `18.000`
- `avg_gang_completion_seconds`: mean `26.000`
- `avg_elastic_wait_seconds`: mean `0.000`
- `avg_elastic_completion_seconds`: mean `0.000`
- `p95_gang_wait_seconds`: mean `18.000`
- `makespan_seconds`: mean `469.455`
- `throughput_jobs_per_minute`: mean `8.180`
- `small_job_bypass_count_while_gang_pending`: mean `0.000`
- `avg_small_wait_seconds`: mean `235.714`

### elastic + BestEffortFIFO

- `head_gang_blocked_seconds`: mean `1.000`
- `avg_gang_wait_seconds`: mean `27.600`
- `avg_gang_completion_seconds`: mean `35.933`
- `avg_elastic_wait_seconds`: mean `40.200`
- `avg_elastic_completion_seconds`: mean `49.200`
- `p95_gang_wait_seconds`: mean `51.400`
- `makespan_seconds`: mean `251.608`
- `throughput_jobs_per_minute`: mean `15.268`
- `small_job_bypass_count_while_gang_pending`: mean `0.000`
- `avg_small_wait_seconds`: mean `121.155`

### elastic + StrictFIFO

- `head_gang_blocked_seconds`: mean `1.000`
- `avg_gang_wait_seconds`: mean `34.467`
- `avg_gang_completion_seconds`: mean `42.800`
- `avg_elastic_wait_seconds`: mean `51.600`
- `avg_elastic_completion_seconds`: mean `60.600`
- `p95_gang_wait_seconds`: mean `64.200`
- `makespan_seconds`: mean `260.472`
- `throughput_jobs_per_minute`: mean `14.751`
- `small_job_bypass_count_while_gang_pending`: mean `0.000`
- `avg_small_wait_seconds`: mean `141.607`

### elastic + learned

- `head_gang_blocked_seconds`: mean `1.000`
- `avg_gang_wait_seconds`: mean `22.333`
- `avg_gang_completion_seconds`: mean `30.493`
- `avg_elastic_wait_seconds`: mean `32.200`
- `avg_elastic_completion_seconds`: mean `42.400`
- `p95_gang_wait_seconds`: mean `37.600`
- `makespan_seconds`: mean `268.904`
- `throughput_jobs_per_minute`: mean `14.196`
- `small_job_bypass_count_while_gang_pending`: mean `0.000`
- `avg_small_wait_seconds`: mean `136.508`

## Pairwise Learned Comparisons

### cohort-best-effort-vs-learned

- `head_gang_blocked_seconds`: `467.400 -> 18.000`, `96.1%` better, `5/5` wins
- `avg_gang_wait_seconds`: `467.400 -> 18.000`, `96.1%` better, `5/5` wins
- `avg_gang_completion_seconds`: `475.400 -> 26.000`, `94.5%` better, `5/5` wins
- `avg_elastic_wait_seconds`: `0.000 -> 0.000`, `0.0%` better, `0/5` wins
- `avg_elastic_completion_seconds`: `0.000 -> 0.000`, `0.0%` better, `0/5` wins
- `p95_gang_wait_seconds`: `467.400 -> 18.000`, `96.1%` better, `5/5` wins
- `makespan_seconds`: `475.582 -> 469.455`, `1.3%` better, `5/5` wins
- `throughput_jobs_per_minute`: `8.074 -> 8.180`, `1.3%` better, `5/5` wins
- `small_job_bypass_count_while_gang_pending`: `62.000 -> 0.000`, `100.0%` better, `5/5` wins
- `avg_small_wait_seconds`: `218.197 -> 235.714`, `-8.0%` better, `0/5` wins

### cohort-strict-vs-learned

- `head_gang_blocked_seconds`: `385.200 -> 18.000`, `95.3%` better, `5/5` wins
- `avg_gang_wait_seconds`: `385.200 -> 18.000`, `95.3%` better, `5/5` wins
- `avg_gang_completion_seconds`: `393.200 -> 26.000`, `93.4%` better, `5/5` wins
- `avg_elastic_wait_seconds`: `0.000 -> 0.000`, `0.0%` better, `0/5` wins
- `avg_elastic_completion_seconds`: `0.000 -> 0.000`, `0.0%` better, `0/5` wins
- `p95_gang_wait_seconds`: `385.200 -> 18.000`, `95.3%` better, `5/5` wins
- `makespan_seconds`: `393.382 -> 469.455`, `-19.4%` better, `0/5` wins
- `throughput_jobs_per_minute`: `9.768 -> 8.180`, `-16.2%` better, `0/5` wins
- `small_job_bypass_count_while_gang_pending`: `62.000 -> 0.000`, `100.0%` better, `5/5` wins
- `avg_small_wait_seconds`: `179.257 -> 235.714`, `-31.5%` better, `0/5` wins

### elastic-best-effort-vs-learned

- `head_gang_blocked_seconds`: `1.000 -> 1.000`, `0.0%` better, `0/5` wins
- `avg_gang_wait_seconds`: `27.600 -> 22.333`, `19.2%` better, `5/5` wins
- `avg_gang_completion_seconds`: `35.933 -> 30.493`, `15.2%` better, `5/5` wins
- `avg_elastic_wait_seconds`: `40.200 -> 32.200`, `19.9%` better, `4/5` wins
- `avg_elastic_completion_seconds`: `49.200 -> 42.400`, `13.8%` better, `4/5` wins
- `p95_gang_wait_seconds`: `51.400 -> 37.600`, `26.8%` better, `5/5` wins
- `makespan_seconds`: `251.608 -> 268.904`, `-6.9%` better, `0/5` wins
- `throughput_jobs_per_minute`: `15.268 -> 14.196`, `-7.0%` better, `0/5` wins
- `small_job_bypass_count_while_gang_pending`: `0.000 -> 0.000`, `0.0%` better, `0/5` wins
- `avg_small_wait_seconds`: `121.155 -> 136.508`, `-12.7%` better, `0/5` wins

### elastic-strict-vs-learned

- `head_gang_blocked_seconds`: `1.000 -> 1.000`, `0.0%` better, `0/5` wins
- `avg_gang_wait_seconds`: `34.467 -> 22.333`, `35.4%` better, `5/5` wins
- `avg_gang_completion_seconds`: `42.800 -> 30.493`, `28.9%` better, `5/5` wins
- `avg_elastic_wait_seconds`: `51.600 -> 32.200`, `37.2%` better, `5/5` wins
- `avg_elastic_completion_seconds`: `60.600 -> 42.400`, `29.7%` better, `5/5` wins
- `p95_gang_wait_seconds`: `64.200 -> 37.600`, `41.6%` better, `5/5` wins
- `makespan_seconds`: `260.472 -> 268.904`, `-3.3%` better, `1/5` wins
- `throughput_jobs_per_minute`: `14.751 -> 14.196`, `-3.7%` better, `1/5` wins
- `small_job_bypass_count_while_gang_pending`: `0.000 -> 0.000`, `0.0%` better, `0/5` wins
- `avg_small_wait_seconds`: `141.607 -> 136.508`, `3.5%` better, `4/5` wins
