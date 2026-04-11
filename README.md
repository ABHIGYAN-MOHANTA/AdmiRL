# AdmiRL

AdmiRL is a research prototype for **Kueue-native admission control for GPU batch workloads**.

It focuses on one question:

**Can a cohort-aware, learning-augmented admission policy improve gang-style and elastic GPU workloads in Kueue under quota borrowing, flavor scarcity, and topology constraints?**

Instead of replacing Kueue, AdmiRL extends the **admission decision layer**:

- `blocked_guard` is the safe hand-written baseline
- `learned_multi_objective` is the PPO-backed runtime policy

The project is built around:

- **Kueue** as the control plane
- **Alibaba Lingjun 2023 traces** as the workload source
- **KWOK + kind** for reproducible local benchmarking
- **PyTorch PPO** for offline policy training