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
