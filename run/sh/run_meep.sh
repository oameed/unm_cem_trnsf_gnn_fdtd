#! /bin/bash

source activate meeppy
cd     run/fdtd

rm     -f  ../../data/meep/*.gif
rm     -rf ../../data/meep/png
mkdir      ../../data/meep/png

echo "RUNNING MEEP EXPERIMENT"
python generator_meep.py  # 42.61 ms

echo "GENERATING GRAPHICS"
mv *.h5    ../../data/meep/png
cd         ../../data/meep/png
h5topng generator_meep-ex.h5 -t 0:398 -R -Zc dkbluered -a yarg -A generator_meep-eps-000000.00.h5 
cd         ../
convert png/generator_*.png ez.gif
xdg-open ez.gif

# this is run on a desktop pc, with Intel(R) Xeon(R) CPU @ 3.20GHz and 32 GB of RAM
