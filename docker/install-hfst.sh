git clone https://github.com/hfst/hfst.git
cd hfst
git checkout 683a5f4e8d457bfa5d25014b7a4346d77427ee20
./autogen.sh
./configure --enable-all-tools --with-unicode-handler=glib
make
make install
