## Quickstart

Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set (optional) environment variables (examples below), then run.

Single problem, validate plan:
```bash
python3 main.py \
  --domain-dir ~/Downloads/pddl-data-master/codmap-2015/unfactored/rovers \
  --domain-file domain \
  --problem-file p10 \
  --mode no-val \
  --validate
```

Batch over all problems in a domain (validated):
```bash
python3 main.py \
  --domain-dir ~/Downloads/pddl-data-master/codmap-2015/unfactored/rovers \
  --domain-file domain \
  --mode no-val \
  --batch \
  --validate
```

Batch over a subset:
```bash
python3 main.py \
  --domain-dir ~/Downloads/pddl-data-master/codmap-2015/unfactored/rovers \
  --domain-file domain \
  --batch \
  --problems p10 p11 p12 \
  --validate
```

Only generate (no validation):
```bash
python3 main.py \
  --domain-dir ~/Downloads/pddl-data-master/codmap-2015/unfactored/rovers \
  --domain-file domain \
  --problem-file p10 \
  --mode no-val
```

Limit plan length:
```bash
python3 main.py \
  --domain-dir ~/Downloads/pddl-data-master/codmap-2015/unfactored/rovers \
  --domain-file domain \
  --problem-file p10 \
  --mode no-val \
  --max-steps 50 \
  --validate
```

Override Validate binary path:
```bash
python3 main.py \
  --domain-dir ~/Downloads/pddl-data-master/codmap-2015/unfactored/rovers \
  --domain-file domain \
  --problem-file p10 \
  --validate \
  --validate-bin /usr/local/bin/Validate
```
