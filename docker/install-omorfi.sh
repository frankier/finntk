git clone https://github.com/frankier/omorfi.git
cd omorfi
git checkout 25-05-2018
./autogen.sh
./configure --enable-labeled-segments --enable-segmenter --enable-hyphenator --enable-lemmatiser --without-java
make
make install