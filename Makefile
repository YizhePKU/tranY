all: split-dataset download-punkt

split-dataset:
	mkdir -p data/interim
	unzip data/raw/conala.zip
	mv conala-corpus/conala-train.json data/interim
	mv conala-corpus/conala-test.json data/
	rm -r conala-corpus
	python3 src/split_dataset.py

download-punkt:
	mkdir -p ~/nltk_data/tokenizers
	cd ~/nltk_data/tokenizers; \
	  wget https://f002.backblazeb2.com/file/sunyizhe/punkt.zip; \
	  unzip punkt.zip; \
	  rm punkt.zip