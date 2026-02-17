nicg_chromatin_builder.py -f chains.txt -o chromatin.pdb -d data.chromatin -b in.bond_settings -hbox 400.0
#export OMP_NUM_THREADS=4 ; lmp -sf hybrid gpu omp -pk omp 4 -pk gpu 1 < in.run
#export OMP_NUM_THREADS=4 ; lmp -sf omp -pk omp 4 < in.run
