#!/usr/bin/env python3
# Writes the builder input for 45 12-nucleosome fibers (30-bp linkers) in a 100 x 100 x 1000 nm slab box:
# fibers/chain_XX.txt (one chain file per fiber), fibers/path_XX.txt (its backbone) and chains.txt (the list).
# The fibers lie along z on a 3 x 3 x 5 lattice. Each backbone is 2 points, the start and the end of the fiber.
import os

nuc      = 12      # nucleosomes per fiber
linker   = 30      # linker DNA (bp) between nucleosomes; no free DNA at the fiber ends
xy       = [-1000./3, 0., 1000./3]               # lattice in x and y (A); the box is 1000 A wide
z        = [-1480., -740., 0., 740., 1480.]      # lattice in z (A); the box is 10000 A long
length   = 600.    # backbone length (A); the built fiber is ~600 A long and ~230 A wide

os.makedirs('fibers', exist_ok=True)
with open('chains.txt', 'w') as chains:
    i = 0
    for x in xy:
        for y in xy:
            for zc in z:
                i += 1
                with open('fibers/path_%02d.txt' % i, 'w') as f:
                    f.write('%.1f %.1f %.1f\n' % (x, y, zc - length/2))
                    f.write('%.1f %.1f %.1f\n' % (x, y, zc + length/2))
                with open('fibers/chain_%02d.txt' % i, 'w') as f:
                    f.write('nuc %d\n' % nuc)
                    f.write('llbp 0' + ' %g' % linker * (nuc - 1) + ' 0\n')
                    f.write('cg fibers/path_%02d.txt\n' % i)
                chains.write('chain fibers/chain_%02d.txt\n' % i)
print('%d fibers written to chains.txt and fibers/' % i)
