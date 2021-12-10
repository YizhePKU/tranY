all: split-train-dev download-punkt

split-train-dev:
	mkdir -p data/interim
	unzip data/raw/conala.zip
	mv conala-corpus/conala-train.json data/interim
	mv conala-corpus/conala-test.json data/
	rm -r conala-corpus
	python3 src/split_train_dev.py

download-punkt:
	mkdir -p ~/nltk_data/tokenizers
	cd ~/nltk_data/tokenizers; \
	  wget https://f002.backblazeb2.com/file/sunyizhe/punkt.zip; \
	  unzip punkt.zip; \
	  rm punkt.zip