nicg_chromatin_builder.py -f chains.txt -o chromatin.pdb -d data.chromatin -b in.bond_settings -hbox 400.0 -twh 10.5 -twN 146 -twC 75 -twsL 1 -t in.dihedral_settings
lmp < in.run
