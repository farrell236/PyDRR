# Benchmark

## Run command

```bash
python -m examples.benchmark_drr input.nii.gz --angle 30 --cpu-cores 8 --cuda-warmup
```

## Machine / environment

```text
- Python 3.9.15
- CuPy 13.6.0
- CPU: AMD EPYC 7543P (32 cores / 64 threads)
- GPU: NVIDIA A100-SXM4-80GB (80 GB)
- NVIDIA Driver Version: 580.126.09
- NVIDIA CUDA Version: 13.0
```


## Benchmark output

```text
Loaded volume
  shape_zyx         : (679, 679, 679)
  spacing_zyx (mm)  : [0.4 0.4 0.4]
  origin_zyx (mm)   : [271.6   0.    0. ]
  direction         :
[[ 1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  0. -1.]]
  intensity range   : [-1000.000, 3095.000]

Isocenter (world xyz mm): [135.6 135.6 407.2]

DRR geometry
  source_mm         : [1001.62537  635.6      407.2    ]
  detector_center   : [-297.4127   -114.399994  407.2     ]
  detector_u        : [-0.5        0.8660254  0.       ]
  detector_v        : [0. 0. 1.]
  detector_size_px  : (512, 512)
  detector_spacing  : (0.51, 0.51)
  detector_size_mm  : (H=261.12, W=261.12)
  source-detector distance (mm): 1500.00
  source-isocenter distance (mm): 1000.00
  iso-detector distance (mm)    : 500.00
  magnification approx          : 1.500
  approx iso-plane FOV mm       : (H=174.08, W=174.08)

[cpu_serial] run 1/1: 250.747 s
cpu_serial stats
  shape             : (512, 512)
  min               : 0.000000
  max               : 61332.128906
  mean              : 12569.330078
  p1 / p99          : 0.000000 / 47844.138477

[cpu_mp] run 1/1: 32.337 s
cpu_mp stats
  shape             : (512, 512)
  min               : 0.000000
  max               : 61332.128906
  mean              : 12569.330078
  p1 / p99          : 0.000000 / 47844.138477

[cuda] warmup render...
[cuda] run 1/1: 0.117 s
cuda stats
  shape             : (512, 512)
  min               : 0.000000
  max               : 61334.765625
  mean              : 12569.354492
  p1 / p99          : 0.000000 / 47846.271094


Benchmark summary
------------------------------------------------------------------------
cpu_serial     mean=250.747s  min=250.747s  max=250.747s  speedup_vs_first=1.00x
cpu_mp         mean=32.337s  min=32.337s  max=32.337s  speedup_vs_first=7.75x
cuda           mean=0.117s  min=0.117s  max=0.117s  speedup_vs_first=2140.80x

Speedup vs cpu_serial
  cpu_serial     1.00x
  cpu_mp         7.75x
  cuda           2140.80x
```

