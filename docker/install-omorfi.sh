set -ex
git clone https://github.com/frankier/omorfi.git
cd omorfi
git checkout rm-homonym-num
./autogen.sh
./configure --enable-labeled-segments --enable-segmenter --enable-hyphenator --enable-lemmatiser --without-java
make
make install
