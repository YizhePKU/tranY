run:
	docker stop clj-python
	docker build -t sunyizhe/clj-python .
	docker run --name clj-python -d --rm --gpus all -p 8848:8848 --env PYTHONPATH=src -v /var/data/sunyz/tranY/data:/workspace/data sunyizhe/clj-python

download-CoNaLa:
	mkdir -p data/raw
	wget --continue --output-document data/raw/conala.zip http://www.phontron.com/download/conala-corpus-v1.1.zip
	
split-train-dev:
	mkdir -p data/interim
	unzip data/raw/conala.zip
	mv conala-corpus/conala-train.json data/interim
	mv conala-corpus/conala-test.json data/
	rm -r conala-corpus
	python3 src/split_train_dev.py

