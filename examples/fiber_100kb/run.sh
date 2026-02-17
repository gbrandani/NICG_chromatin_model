nicg_define_fiber.py --config settings.input.json
nicg_chromatin_builder.py -f chains.txt -o chromatin.pdb -d data.chromatin -b in.bond_settings -hbox 800
lmp < in.relax
lmp < in.run
