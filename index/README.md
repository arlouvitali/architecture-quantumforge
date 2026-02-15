# FAISS Index

Индекс строится скриптом `scripts/build_index.py`.

## Сборка (запуск из корня репозитория `architecture-quantumforge`)

```bash
. .venv/bin/activate
python scripts/build_index.py --run-sample-queries
```

Офлайн-режим (если модель уже в локальном кэше):

```bash
. .venv/bin/activate
python scripts/build_index.py --run-sample-queries --local-files-only
```

## Выходные файлы

- `faiss_index/index.faiss`
- `faiss_index/index.pkl`
- `faiss_index/index_stats.json`
- `faiss_index/sample_query_results.md`
