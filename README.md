# tranY

## Development Setup

**Attention: Python 3.9 (or later) is required.**

1. Clone the repository. The CoNaLa dataset is included in the repo(~50MB) so this may take several minutes.

```bash
git clone git@github.com:YizhePKU/tranY.git
```

2. Install dependencies.

```bash
python3 -m pip install -r requirements.txt
```

3. Source files are located inside `src/`, so you need to add it to Python search path. Consider adding it to your `.bashrc` so that you don't have to type it every time.

```bash
export PYTHONPATH=src
```

4. Unzip the CoNaLa dataset and split the training data into `train` and `dev` dataset.

```bash
make split-train-dev
```

5. (Optional) run the test suite.

```bash
pytest
```

6. Train the model.

```
python3 src/main.py
```

## Todos

- [ ] add attention to the model
- [ ] add parent feeding to the model
- [ ] save checkpoints of trained model
- [ ] evaluate using BLEU
