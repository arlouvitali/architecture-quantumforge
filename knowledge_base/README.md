# Knowledge Base

- `raw/` — исходные документы по миру Star Wars (34 файла).
- `final/` — документы после подмены терминов (34 файла).

## Замены терминов

Словарь замен хранится в файле `../terms_map.json`.

## Как пересобрать `final/`

```bash
python3 ../scripts/replace_terms.py \
  --input-dir ./raw \
  --output-dir ./final \
  --map ../terms_map.json
```
