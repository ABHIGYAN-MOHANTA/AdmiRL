# AdmiRL Grafana Bundle

This folder contains a small Grafana + Prometheus bundle for the AdmiRL model
server.

What it expects:

- the model server is running on the host at `http://127.0.0.1:5050`
- the model server exposes Prometheus-format metrics at `/metrics`

What is included:

- `docker-compose.yml`
  - starts Prometheus and Grafana locally
- `prometheus/prometheus.yml`
  - scrapes `host.docker.internal:5050/metrics`
- `provisioning/datasources/prometheus.yaml`
  - auto-adds the Prometheus datasource
- `provisioning/dashboards/dashboards.yaml`
  - auto-loads dashboards from `dashboards/`
- `dashboards/admirl-runtime-overview.json`
  - dashboard focused on latency, decision source mix, and runtime gauges

Start it with:

```bash
cd model_server/grafana
docker compose up -d
```

Then open:

- Grafana: `http://127.0.0.1:3000`
- Prometheus: `http://127.0.0.1:9090`

Default Grafana login:

- user: `admin`
- password: `admin`

If you are not on Docker Desktop / macOS, you may need to replace
`host.docker.internal` in `prometheus/prometheus.yml` with a host-reachable
address for the machine running the AdmiRL server.
