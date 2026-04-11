# Kueue Benchmark Harness

This directory contains the new distributed-training benchmark path built
around:

- Alibaba Lingjun 2023 trace replay
- Kueue `ClusterQueue` / `LocalQueue` / `ResourceFlavor`
- plain pod groups for gang-style workloads
- a Kueue fork checkout, resolved from GitHub by default and overridable for
  local experiments

Main entrypoints:

- `python3 test/kueue/run_kueue_benchmarks.py`
- `python3 test/kueue/provisioner.py`

Generated artifacts include:

- `setup.yaml`: topology, flavors, queues, admission checks
- `workloads/*.yaml`: one plain pod group per Lingjun-derived workload
- `meta.json`: arrival order and workload metadata for the benchmark arm

Notes:

- By default, the live runner clones `https://github.com/ABHIGYAN-MOHANTA/kueue.git`
  at ref `main` into a cache directory under `/tmp`.
- `ADMIRL_KUEUE_SOURCE_DIR` overrides the source tree directly.
- `ADMIRL_KUEUE_SOURCE_MODE=local` forces the runner to use
  `/Users/abhigyan/Desktop/kueue` as a local fallback checkout.
- KWOK nodes should be generated with
  `python3 test/kwok/generate_fake_nodes.py --layout training-balanced`
  or another training layout so their labels match the generated
  `ResourceFlavor` objects.
