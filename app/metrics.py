from prometheus_client import Counter, Gauge, Histogram

# How long each inference takes — tells you if the model is getting slow
inference_duration_seconds = Histogram(
    "gemma_inference_duration_seconds",
    "Model inference wall-clock time",
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60, 120],
)

# Tokens in vs tokens out — tells you throughput
prompt_tokens_total = Counter(
    "gemma_prompt_tokens_total",
    "Cumulative prompt tokens received",
)
completion_tokens_total = Counter(
    "gemma_completion_tokens_total",
    "Cumulative completion tokens generated",
)

# Tokens per second derived from duration + completion count
tokens_per_second = Histogram(
    "gemma_tokens_per_second",
    "Completion tokens generated per second",
    buckets=[1, 5, 10, 20, 30, 50, 75, 100, 150],
)

# How many requests are in-flight right now
active_requests = Gauge(
    "gemma_active_requests",
    "Requests currently being processed by the model",
)

# Request outcomes
requests_total = Counter(
    "gemma_requests_total",
    "Total requests by endpoint and outcome",
    ["status", "endpoint"],  # endpoint: analyze | generate
)

# 1 when model is loaded, 0 when not
model_loaded_gauge = Gauge(
    "gemma_model_loaded",
    "Whether the model is loaded and ready (1=yes, 0=no)",
)

# Requests waiting to acquire a concurrency slot
queued_requests = Gauge(
    "gemma_queued_requests",
    "Requests waiting in queue for a concurrency slot",
)

# All-time analyze success count — initialized from CSV on startup so it
# survives container restarts without resetting to 0
analyze_success_total = Gauge(
    "gemma_analyze_success_total",
    "All-time successful /analyze calls (persisted via CSV, survives restarts)",
)

# All-time translate success count — initialized from /data/translate_count.txt
translate_success_total = Gauge(
    "gemma_translate_success_total",
    "All-time successful /translate calls (persisted via file, survives restarts)",
)

# Dedicated histogram for /translate inference duration — separate from the
# shared histogram so p50/p95/p99 can be queried per-endpoint in Grafana
translate_duration_seconds = Histogram(
    "gemma_translate_duration_seconds",
    "Inference wall-clock time for /translate calls",
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60, 120],
)

# Dedicated histogram for /vision inference duration
vision_duration_seconds = Histogram(
    "gemma_vision_duration_seconds",
    "Inference wall-clock time for /vision calls",
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60, 120],
)
