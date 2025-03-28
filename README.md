# Fast Multipole Method (FMM) for Collisional N-body Simulations

This repository contains an implementation of the Fast Multipole Method (FMM) as described [here](https://comp-astrophys-cosmol.springeropen.com/articles/10.1186/s40668-014-0001-7). The code is designed for high-performance collisional N-body simulations in astrophysics, where efficient computation of gravitational forces is critical. It leverages multipole expansions and spherical harmonics to accurately and efficiently approximate gravitational interactions.

## Reproduction
1. **Prerequisites:**
```bash
git clone https://github.com/IIITyforIII/FMM #clone reo
```

1.1: Using uv (recommended:)

1.2 Python:
```bash
pip install -r requirements.txt #...
```

## Visualizations
1. Volume Rendering of a Point Cloud Density Map
2. Point Cloud Visualization
3. Time Series Animation

## Modules
### geolib
- Tree implementation based on [here](https://ieeexplore.ieee.org/document/6495868)
### fmm
- kernels.py
### simlib
- available simulators
### utils
- performancetest
- visualization.py
### physlib
- create different environments


## Structure
```bash
├── src
│   ├── fmm
│   │   ├── __init__.py
│   │   └── kernels.py
│   ├── geolib
│   │   ├── __init__.py
│   │   ├── cell.py
│   │   ├── coordinates.py
│   │   ├── expansionCentres.py
│   │   ├── oct_tree.py
│   │   └── tree.py
│   ├── main.py
│   ├── physlib
│   │   ├── __init__.py
│   │   ├── densityModels.py
│   │   └── entities.py
│   ├── simlib
│   │   ├── __init__.py
│   │   ├── acceptanceCriterion.py
│   │   ├── kernels.py
│   │   └── simulators.py
│   ├── simulationBase.py
│   └── utils
│       ├── __init__.py
│       ├── dataIO.py
│       ├── heatmap.py
│       ├── performanceTest.py
│       └── visualization.py
├── traceData
│   └── trace1.npy
└── uv.lock
├── README.md
├── data
├── heatmapVideos
├── praesentation
│   └── structure.txt
├── pyproject.toml
├── requirements-cuda12.txt
├── requirements.txt
```