git clone https://github.com/mpsilfve/FinnPos/
cd FinnPos
git checkout 82bce745afc658444fab1bb58b47bff89c207ba2
make
wget https://github.com/mpsilfve/FinnPos/releases/download/v0.1-alpha/morphology.omor.hfst.gz
gunzip morphology.omor.hfst.gz
mv morphology.omor.hfst share/finnpos/omorfi/morphology.omor.hfst
make ftb-omorfi-tagger
make install
make install-models
