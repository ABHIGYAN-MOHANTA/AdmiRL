# Alibaba Lingjun 2023 Trace

This folder contains the local working copy of the public **Alibaba GPU Cluster
Dataset 2023**:

- Source repo:
  [alibaba/alibaba-lingjun-dataset-2023](https://github.com/alibaba/alibaba-lingjun-dataset-2023)
- Paper:
  *Crux: GPU-Efficient Communication Scheduling for Deep Learning Training*
  (ACM SIGCOMM 2024)

## Why This Dataset

This dataset fits the gang-scheduling direction well because it contains:

- **job-level metadata** in [`job.csv`](../../data/alibaba_lingjun_2023/job.csv)
- **worker-level replica information** in [`worker.csv`](../../data/alibaba_lingjun_2023/worker.csv)
- **network topology metadata** in [`topo.csv`](../../data/alibaba_lingjun_2023/topo.csv)

That makes it useful for:

- gang-style or worker-group admission
- topology-aware placement
- heterogeneous accelerator scheduling
- queue-aware RL for distributed training jobs

## Local Snapshot

Current local row counts:

- `job.csv`: `5,180`
- `worker.csv`: `23,743`
- `topo.csv`: `848`

## Important Columns

`job.csv` includes fields such as:

- `job_name`
- `kind`
- `model`
- `priority`
- `gmt_job_submitted`
- `gmt_job_running`
- `gmt_job_finished`

`worker.csv` includes fields such as:

- `job_name`
- `worker_name`
- `replica_type`
- `host_ip`
- `RES`

`topo.csv` includes fields such as:

- `ip`
- `DSW`
- `PSW`
- `ASW`

## Notes

- The CSV files are treated as local data inputs for development.
- The upstream repository remains the source of truth.
- This folder is the canonical starting point for the upcoming Kueue /
  gang-scheduling implementation.
