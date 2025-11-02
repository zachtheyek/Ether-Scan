# Database Module - Aetherium

This directory contains all database-related code for the Aetherscan pipeline - Aetherium metrics collection system.

## Files

- **`__init__.py`**: Package initialization that exports public functions
- **`db.py`**: Core SQLite database management with queue-based writer
- **`helpers.py`**: Convenience functions for logging metrics from any process
- **`analyze.py`**: Standalone script for analyzing metrics after a run

## Usage

### Import from the db package:

```python
# In main.py or other pipeline files
from db import init_db, get_db, shutdown_db

# In worker processes (data_generation.py, training.py)
from db import log_data_generation_start, log_data_generation_complete
from db import log_training_epoch_metric
```

### Run analysis script:

```bash
python db/analyze.py /path/to/aetherium.db
python db/analyze.py /path/to/aetherium.db --plot output.png
```

## Architecture

The Aetherium system uses a queue-based architecture for safe multiprocess writes:

1. All processes → Shared Queue → Single Writer Thread → SQLite (Aetherium)
2. No concurrent write issues
3. Non-blocking metric logging
4. Automatic batching every 5 seconds or 100 records

See `../METRICS_USAGE.md` for complete documentation.


# Aetherium Database Usage Guide

This guide explains how to use the SQLite-based data collection system (hereafter: the Aetherium database) in the Aetherscan pipeline.

## Architecture

The Aetherium database uses a **queue-based architecture** (similar to the logging system):

```
┌─────────────────────────────────────────────────────────────┐
│  Main Process                                               │
│  ┌──────────────────┐         ┌────────────────────┐        │
│  │ Resource Monitor │────────▶│ Writer Queue       │        │
│  └──────────────────┘         │ (thread-safe)      │        │
│                               └────────────────────┘        │
│  ┌──────────────────┐                  │                    │
│  │ Training Code    │──────────────────┤                    │
│  └──────────────────┘                  │                    │
│                                        ▼                    │
│                               ┌────────────────────┐        │
│                               │ Writer Thread      │        │
│                               │ (single writer)    │        │
│                               └────────────────────┘        │
│                                        │                    │
│                                        ▼                    │
│                               ┌────────────────────┐        │
│                               │ Aetherium Database │        │
│                               │ (metrics.db)       │        │
│                               └────────────────────┘        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Worker Processes (data_generation.py workers)              │
│  ┌──────────────────┐         ┌────────────────────┐       │
│  │ Signal Injection │────────▶│  Metrics Queue     │───────┼──▶ (shared queue)
│  └──────────────────┘         │  (inherited)       │       │
└─────────────────────────────────────────────────────────────┘
```

### Key Benefits

1. **No locking issues**: Single writer thread eliminates `SQLITE_BUSY` errors
2. **Non-blocking**: All metric writes go to queue, never block computation
3. **Multiprocess safe**: Worker processes send to queue, main process writes
4. **Minimal memory**: Only small rolling buffer kept in RAM
5. **Persistent**: All data saved to disk, survives crashes

## Database Schema

### `resource_metrics` table
- System resource monitoring (CPU, RAM, GPU)
- Automatically logged by `ResourceMonitor` class
- 1 sample per second by default

### `generation_metrics` table
- Data generation performance metrics
- Log from worker processes in `data_generation.py`
- Tracks: duration, samples generated, memory usage

### `training_metrics` table
- Training loop metrics (loss, accuracy, etc.)
- Log from training code
- Tracks: epoch metrics, learning rate, validation scores

## Usage Examples

### 1. Resource Monitoring (Automatic)

Resource monitoring is automatically initialized in `train_command()`:

```python
# In main.py train_command() - already implemented
init_metrics_db(db_path, buffer_size=100)
log_system_resources(output_path)

# ResourceMonitor automatically logs:
# - CPU usage (system-wide and process tree)
# - RAM usage (system-wide and process tree)
# - GPU usage and memory (per GPU)
```

### 2. Data Generation Metrics (Manual)

Add metrics logging to data generation operations:

```python
# In data_generation.py
from db import log_data_generation_start, log_data_generation_complete

# At the start of batch_create_cadence()
def batch_create_cadence(function, samples, plate, ...):
    start_time = log_data_generation_start(
        operation='signal_injection',
        round_number=current_round,  # pass from caller
        chunk_number=current_chunk   # pass from caller
    )

    # ... existing generation code ...

    # At the end
    log_data_generation_complete(
        start_time=start_time,
        operation='signal_injection',
        samples_generated=samples,
        round_number=current_round,
        chunk_number=current_chunk
    )
```

### 3. Training Metrics (Manual)

Log training metrics from your training loop:

```python
# In training.py
from db import log_training_epoch_metric

# After each epoch
for epoch in range(epochs):
    # ... training code ...

    # Log training loss
    log_training_epoch_metric(
        metric_name='loss',
        metric_value=train_loss,
        round_number=current_round,
        epoch_number=epoch,
        phase='train'
    )

    # Log validation loss
    log_training_epoch_metric(
        metric_name='loss',
        metric_value=val_loss,
        round_number=current_round,
        epoch_number=epoch,
        phase='val'
    )

    # Log learning rate
    log_training_epoch_metric(
        metric_name='learning_rate',
        metric_value=current_lr,
        round_number=current_round,
        epoch_number=epoch,
        phase='train'
    )
```

## Querying Metrics

### From Python

```python
from db import get_metrics_db

metrics_db = get_metrics_db()

# Query resource metrics
cpu_metrics = metrics_db.query_resource_metrics(metric_type='cpu')

# Get summary statistics
stats = metrics_db.get_summary_stats()
print(f"Total samples: {stats['resource_metrics_count']}")
print(f"Duration: {stats['duration_minutes']:.2f} minutes")
print(f"DB size: {stats['db_size_mb']:.2f} MB")
```

### From SQL (Direct)

```bash
# Connect to the database
sqlite3 /path/to/metrics.db

# Query average CPU usage
SELECT
    AVG(value) as avg_cpu,
    MAX(value) as max_cpu,
    MIN(value) as min_cpu
FROM resource_metrics
WHERE metric_type = 'cpu' AND metric_name = 'system_total';

# Query data generation performance by round
SELECT
    round_number,
    COUNT(*) as num_operations,
    AVG(duration_seconds) as avg_duration,
    SUM(samples_generated) as total_samples,
    AVG(memory_used_mb) as avg_memory_mb
FROM generation_metrics
WHERE operation = 'signal_injection'
GROUP BY round_number
ORDER BY round_number;

# Query training loss over time
SELECT
    round_number,
    epoch_number,
    phase,
    metric_value as loss
FROM training_metrics
WHERE metric_name = 'loss'
ORDER BY round_number, epoch_number, phase;
```

## Integration with data_generation.py

Here's a complete example of how to add metrics to `DataGenerator.generate_train_batch()`:

```python
# In data_generation.py DataGenerator class

def generate_train_batch(self, n_samples: int, snr_base: int, snr_range: int,
                        round_number: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Generate training batch with metrics logging"""

    from db import log_data_generation_start, log_data_generation_complete
    import psutil

    max_chunk_size = self.config.training.signal_injection_chunk_size
    n_chunks = max(1, (n_samples + max_chunk_size - 1) // max_chunk_size)

    logger.info(f"Generating {n_samples} samples in {n_chunks} chunks of max {max_chunk_size}")

    # Pre-allocate output arrays
    all_main = np.empty((n_samples, 6, 16, self.width_bin), dtype=np.float32)
    all_false = np.empty((n_samples, 6, 16, self.width_bin), dtype=np.float32)
    all_true = np.empty((n_samples, 6, 16, self.width_bin), dtype=np.float32)

    for chunk_idx in range(n_chunks):
        # Track memory at chunk start
        process = psutil.Process(os.getpid())
        mem_start_mb = process.memory_info().rss / (1024 * 1024)

        chunk_size = min(max_chunk_size, n_samples - chunk_idx * max_chunk_size)
        if chunk_size <= 0:
            break

        start_idx = chunk_idx * max_chunk_size
        end_idx = start_idx + chunk_size

        # Log chunk start
        chunk_start_time = log_data_generation_start(
            operation='signal_injection_chunk',
            round_number=round_number,
            chunk_number=chunk_idx
        )

        logger.info(f"Generating chunk {chunk_idx + 1}/{n_chunks} with {chunk_size} samples")

        # ... existing generation code (quarter splits, batch_create_cadence calls) ...

        # Track memory at chunk end
        mem_end_mb = process.memory_info().rss / (1024 * 1024)
        mem_used_mb = mem_end_mb - mem_start_mb

        # Log chunk completion
        log_data_generation_complete(
            start_time=chunk_start_time,
            operation='signal_injection_chunk',
            samples_generated=chunk_size,
            round_number=round_number,
            chunk_number=chunk_idx,
            memory_used_mb=mem_used_mb
        )

        logger.info(f"Chunk {chunk_idx + 1} complete, memory cleared")

    return result
```

## Performance Considerations

### Memory Usage

- **Old approach**: All data in RAM (~1.5 GB for 24-hour run at 1 Hz)
- **New approach**: 60-second rolling buffer (~5 KB) + SQLite (~50 MB for 24-hour run)
- **Savings**: ~97% reduction in memory usage

### Write Performance

- Batched writes every 5 seconds or 100 records (whichever comes first)
- ~200 metrics/sec with negligible CPU overhead (<0.1%)
- WAL mode enabled for better concurrent read performance

### Database Size Estimates

For a typical training run:
- Resource metrics: ~3.6 million records/day (6 metrics × 1 Hz × 86400 sec)
- Database size: ~50-100 MB/day (compressed with indexes)
- Query performance: <100ms for typical aggregations

## Why SQLite Over PostgreSQL?

### SQLite Advantages for HPC

1. **No sudo required**: Just a file, no server installation
2. **No network**: No port conflicts on HPC job arrays
3. **Single-process optimal**: Perfect for our use case
4. **Portable**: Copy database file between systems easily
5. **Zero configuration**: No connection strings, authentication, etc.

### When to Consider PostgreSQL

Only if you need:
- Centralized monitoring across multiple HPC jobs simultaneously
- Real-time complex queries during execution
- Multiple concurrent writers (we have single writer via queue)
- Monitoring dashboard for live cluster-wide metrics

For single-job monitoring (even long-running ones), **SQLite is the optimal choice**.

## Troubleshooting

### "Metrics database not initialized"
- Ensure `init_metrics_db()` is called before `log_system_resources()`
- Check that you're in the main process (not a worker)

### "SQLITE_BUSY" errors
- Should never happen with queue-based architecture
- If it does, check that you're using `log_*` functions, not writing directly

### Missing metrics in plot
- Check database with: `sqlite3 metrics.db "SELECT COUNT(*) FROM resource_metrics;"`
- Ensure `shutdown_metrics_db()` is called to flush final writes
- Check logs for exceptions in writer thread

### Database file too large
- Reduce buffer_size in `init_metrics_db()` (default: 100)
- Increase monitoring interval in `ResourceMonitor` (default: 1.0 sec)
- Run `VACUUM` to reclaim space: `sqlite3 metrics.db "VACUUM;"`
