# Tuned 5-Seed Kueue Suite

## Arm Means

### cohort + BestEffortFIFO

- `head_gang_blocked_seconds`: mean `52.400`
- `avg_gang_wait_seconds`: mean `52.400`
- `avg_gang_completion_seconds`: mean `60.400`
- `avg_elastic_wait_seconds`: mean `0.000`
- `avg_elastic_completion_seconds`: mean `0.000`
- `p95_gang_wait_seconds`: mean `52.400`
- `makespan_seconds`: mean `61.600`
- `throughput_jobs_per_minute`: mean `7.798`
- `small_job_bypass_count_while_gang_pending`: mean `6.000`
- `avg_small_wait_seconds`: mean `12.514`

### cohort + StrictFIFO

- `head_gang_blocked_seconds`: mean `53.200`
- `avg_gang_wait_seconds`: mean `53.200`
- `avg_gang_completion_seconds`: mean `61.200`
- `avg_elastic_wait_seconds`: mean `0.000`
- `avg_elastic_completion_seconds`: mean `0.000`
- `p95_gang_wait_seconds`: mean `53.200`
- `makespan_seconds`: mean `62.400`
- `throughput_jobs_per_minute`: mean `7.693`
- `small_job_bypass_count_while_gang_pending`: mean `6.000`
- `avg_small_wait_seconds`: mean `12.686`

### cohort + learned

- `head_gang_blocked_seconds`: mean `12.600`
- `avg_gang_wait_seconds`: mean `12.600`
- `avg_gang_completion_seconds`: mean `20.600`
- `avg_elastic_wait_seconds`: mean `0.000`
- `avg_elastic_completion_seconds`: mean `0.000`
- `p95_gang_wait_seconds`: mean `12.600`
- `makespan_seconds`: mean `61.600`
- `throughput_jobs_per_minute`: mean `7.793`
- `small_job_bypass_count_while_gang_pending`: mean `0.000`
- `avg_small_wait_seconds`: mean `25.171`

### elastic + BestEffortFIFO

- `head_gang_blocked_seconds`: mean `1.200`
- `avg_gang_wait_seconds`: mean `16.150`
- `avg_gang_completion_seconds`: mean `24.450`
- `avg_elastic_wait_seconds`: mean `32.000`
- `avg_elastic_completion_seconds`: mean `41.000`
- `p95_gang_wait_seconds`: mean `34.000`
- `makespan_seconds`: mean `43.270`
- `throughput_jobs_per_minute`: mean `11.094`
- `small_job_bypass_count_while_gang_pending`: mean `0.000`
- `avg_small_wait_seconds`: mean `3.300`

### elastic + StrictFIFO

- `head_gang_blocked_seconds`: mean `1.200`
- `avg_gang_wait_seconds`: mean `13.500`
- `avg_gang_completion_seconds`: mean `21.750`
- `avg_elastic_wait_seconds`: mean `29.600`
- `avg_elastic_completion_seconds`: mean `38.600`
- `p95_gang_wait_seconds`: mean `31.800`
- `makespan_seconds`: mean `42.941`
- `throughput_jobs_per_minute`: mean `11.181`
- `small_job_bypass_count_while_gang_pending`: mean `0.000`
- `avg_small_wait_seconds`: mean `14.550`

### elastic + learned

- `head_gang_blocked_seconds`: mean `1.000`
- `avg_gang_wait_seconds`: mean `5.667`
- `avg_gang_completion_seconds`: mean `13.667`
- `avg_elastic_wait_seconds`: mean `14.600`
- `avg_elastic_completion_seconds`: mean `25.600`
- `p95_gang_wait_seconds`: mean `14.200`
- `makespan_seconds`: mean `33.919`
- `throughput_jobs_per_minute`: mean `14.151`
- `small_job_bypass_count_while_gang_pending`: mean `0.000`
- `avg_small_wait_seconds`: mean `13.280`

## Pairwise Learned Comparisons

### cohort-best-effort-vs-learned

- `head_gang_blocked_seconds`: `52.400 -> 12.600`, `75.9%` better, `5/5` wins
- `avg_gang_wait_seconds`: `52.400 -> 12.600`, `75.9%` better, `5/5` wins
- `avg_gang_completion_seconds`: `60.400 -> 20.600`, `65.9%` better, `5/5` wins
- `avg_elastic_wait_seconds`: `0.000 -> 0.000`, `0.0%` better, `0/5` wins
- `avg_elastic_completion_seconds`: `0.000 -> 0.000`, `0.0%` better, `0/5` wins
- `p95_gang_wait_seconds`: `52.400 -> 12.600`, `75.9%` better, `5/5` wins
- `makespan_seconds`: `61.600 -> 61.600`, `-0.1%` better, `3/5` wins
- `throughput_jobs_per_minute`: `7.798 -> 7.793`, `0.0%` better, `3/5` wins
- `small_job_bypass_count_while_gang_pending`: `6.000 -> 0.000`, `100.0%` better, `5/5` wins
- `avg_small_wait_seconds`: `12.514 -> 25.171`, `-101.3%` better, `0/5` wins

### cohort-strict-vs-learned

- `head_gang_blocked_seconds`: `53.200 -> 12.600`, `76.3%` better, `5/5` wins
- `avg_gang_wait_seconds`: `53.200 -> 12.600`, `76.3%` better, `5/5` wins
- `avg_gang_completion_seconds`: `61.200 -> 20.600`, `66.3%` better, `5/5` wins
- `avg_elastic_wait_seconds`: `0.000 -> 0.000`, `0.0%` better, `0/5` wins
- `avg_elastic_completion_seconds`: `0.000 -> 0.000`, `0.0%` better, `0/5` wins
- `p95_gang_wait_seconds`: `53.200 -> 12.600`, `76.3%` better, `5/5` wins
- `makespan_seconds`: `62.400 -> 61.600`, `1.3%` better, `5/5` wins
- `throughput_jobs_per_minute`: `7.693 -> 7.793`, `1.3%` better, `5/5` wins
- `small_job_bypass_count_while_gang_pending`: `6.000 -> 0.000`, `100.0%` better, `5/5` wins
- `avg_small_wait_seconds`: `12.686 -> 25.171`, `-98.5%` better, `0/5` wins

### elastic-best-effort-vs-learned

- `head_gang_blocked_seconds`: `1.200 -> 1.000`, `10.0%` better, `1/5` wins
- `avg_gang_wait_seconds`: `16.150 -> 5.667`, `64.8%` better, `5/5` wins
- `avg_gang_completion_seconds`: `24.450 -> 13.667`, `44.0%` better, `5/5` wins
- `avg_elastic_wait_seconds`: `32.000 -> 14.600`, `53.4%` better, `5/5` wins
- `avg_elastic_completion_seconds`: `41.000 -> 25.600`, `36.8%` better, `5/5` wins
- `p95_gang_wait_seconds`: `34.000 -> 14.200`, `58.2%` better, `5/5` wins
- `makespan_seconds`: `43.270 -> 33.919`, `21.6%` better, `5/5` wins
- `throughput_jobs_per_minute`: `11.094 -> 14.151`, `27.6%` better, `5/5` wins
- `small_job_bypass_count_while_gang_pending`: `0.000 -> 0.000`, `0.0%` better, `0/5` wins
- `avg_small_wait_seconds`: `3.300 -> 13.280`, `-366.1%` better, `0/5` wins

### elastic-strict-vs-learned

- `head_gang_blocked_seconds`: `1.200 -> 1.000`, `10.0%` better, `1/5` wins
- `avg_gang_wait_seconds`: `13.500 -> 5.667`, `57.6%` better, `5/5` wins
- `avg_gang_completion_seconds`: `21.750 -> 13.667`, `36.9%` better, `5/5` wins
- `avg_elastic_wait_seconds`: `29.600 -> 14.600`, `42.6%` better, `4/5` wins
- `avg_elastic_completion_seconds`: `38.600 -> 25.600`, `28.6%` better, `4/5` wins
- `p95_gang_wait_seconds`: `31.800 -> 14.200`, `54.3%` better, `5/5` wins
- `makespan_seconds`: `42.941 -> 33.919`, `21.0%` better, `5/5` wins
- `throughput_jobs_per_minute`: `11.181 -> 14.151`, `26.6%` better, `5/5` wins
- `small_job_bypass_count_while_gang_pending`: `0.000 -> 0.000`, `0.0%` better, `0/5` wins
- `avg_small_wait_seconds`: `14.550 -> 13.280`, `0.7%` better, `4/5` wins
