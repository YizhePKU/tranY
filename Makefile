split-train-dev:
	mkdir -p data/interim
	unzip data/raw/conala.zip
	mv conala-corpus/conala-train.json data/interim
	mv conala-corpus/conala-test.json data/
	rm -r conala-corpus
	python3 src/split_train_dev.py
