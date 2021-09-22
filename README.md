# tranY

## Development Setup

**Attention: Python 3.9 (or later) is required.**

1. Clone the repository:

```bash
git clone git@github.com:YizhePKU/tranY.git
```

2. Install dependencies:

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
