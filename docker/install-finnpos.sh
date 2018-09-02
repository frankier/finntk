git clone https://github.com/mpsilfve/FinnPos/
cd FinnPos
git checkout 82bce745afc658444fab1bb58b47bff89c207ba2
cp ../finnpos-makefile.patch .
patch Makefile finnpos-makefile.patch
make
wget https://github.com/mpsilfve/FinnPos/releases/download/v0.1-alpha/morphology.omor.hfst.gz
gunzip morphology.omor.hfst.gz
mv morphology.omor.hfst share/finnpos/omorfi/morphology.omor.hfst
wget https://github.com/frankier/finntk/releases/download/bins/ftb_omorfi_model.zip
unzip ftb_omorfi_model.zip
make install
make install-models
