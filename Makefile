CoNaLa:
	wget --continue --output-document data/raw/conala.zip http://www.phontron.com/download/conala-corpus-v1.1.zip
	unzip data/raw/conala.zip
	mv conala-corpus/* data/
	rmdir conala-corpus