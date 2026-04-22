#! /bin/bash


#PYTHONPATH=. python examples/demo_phase1.py \
#  /data/houbb/data/NIDCR_data/CARSH/CARSH_00073_0000.nii.gz \
#  --invert \
#  --n-cores 8 \
#  --mp-chunksize 2

#PYTHONPATH=. python examples/demo_phase2_pano.py \
#  /data/houbb/data/NIDCR_data/CARSH/CARSH_00073_0000.nii.gz \
#  -o pano_stage2.png \
#  --angle-start -45 \
#  --angle-end 45 \
#  --n-views 121 \
#  --det-h 384 \
#  --det-w 384 \
#  --strip-width 1 \
#  --columns-per-view 1 \
#  --reduce center \
#  --center-col-start 120 \
#  --center-col-end 260 \
#  --threshold -900 \
#  --invert \
#  --n-cores 8 --mp-chunksize 2


PYTHONPATH=. python examples/demo_pano_rebuild_v2.py \
  /data/houbb/data/NIDCR_data/CARSH/CARSH_00073_0000.nii.gz \
  -o pano_rebuild_v2.png \
  --n-points 81 \
  --patch-h 256 \
  --patch-w 9 \
  --columns-per-state 3 \
  --reduce gaussian \
  --gaussian-sigma 1.0 \
  --row-start-frac 0.50 \
  --row-end-frac 0.92 \
  --trough-sigma-mm 12 \
  --z-sigma-mm 18 \
  --threshold -900 \
  --invert \
  --n-cores 32


#python get_drr_siddon_jacobs.py \
#  /data/houbb/data/NIDCR_data/CARSH/CARSH_00073_0000.nii.gz \
#  -res 0.51 0.51 \
#  -size 512 512 \
#  -scd 1000 \
#  -rp 90 \
#  -threshold -900 \
#  --invert \
#  --n-cores 8 \
#  --mp-chunksize 2 \
#  -o drr_mp.png
