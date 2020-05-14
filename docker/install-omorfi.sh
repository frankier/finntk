set -ex
git clone --depth 1 https://github.com/frankier/omorfi.git --branch rm-homonym-num --single-branch
cd omorfi
./autogen.sh
./configure --enable-labeled-segments --enable-segmenter --enable-hyphenator --enable-lemmatiser --without-java
make
make install
