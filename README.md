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

3. Source files are located inside `src/`, so you need to add it to Python search path.

```bash
export PYTHONPATH=src
```
 
 Alternatively, you can add the following lines to your entry scripts:
 
```python3
import sys
sys.path.append("src")

# rest of your script...
```

 This is useful when it's hard to modify `PYTHONPATH` for some reason, e.g. when running a Jupyter Notebook.

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

 Make sure your working directory is at the project root (instead of `src/`).

## Todos

- [x] save checkpoints of trained model
- [x] add parent feeding to the model
- [ ] add attention to the model
- [ ] evaluate using BLEU
- [ ] add Transformer as an alternative to LSTM
