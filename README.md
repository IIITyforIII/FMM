# Fast Multipole Method for N-body Simulations

This repository contains an implementation of the Fast Multipole Method (FMM) as described by Dehnen ([2014](https://comp-astrophys-cosmol.springeropen.com/articles/10.1186/s40668-014-0001-7)). The code is designed for high-performance collisional N-body simulations in astrophysics, where efficient computation of gravitational forces is critical. It leverages multipole expansions and spherical harmonics to accurately and efficiently approximate gravitational interactions.


## Setup 

```bash
git clone https://github.com/IIITyforIII/FMM.git
cd FMM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt # or requriements-cuda12.txt (for GPU acceleration) 
```

## Run simulation
```bash
python src/simulationBase.py
```


## Supported Visualizations (see `src/utils/visualization.py`)
1. Volume Rendering of a Point Cloud Density Map
2. Point Cloud Visualization
3. Time Series Animation

## Modules

### geolib (`src/geolib`)

- **Tree Implementation:** Based on Taura et al. ([2012](https://ieeexplore.ieee.org/document/6495868)).
- **cell.py:** Contains the cell class.
- **tree.py:** Implements an oct-tree for spatial partitioning using Morton sorting, node splitting, periodic boundary handling, and optional multi-threaded particle insertion.
- **expansionCentres.py:** Provides an interface for computing expansion centers in tree nodes.

### physlib (`src/physlib`)

- **densityModels.py:** Density model classes (UniformBox, UniformSphere, and PlummerSphere) for sampling particle positions and velocities.
- **entities.py:** Defines Particle and Star classes for representing point masses in 3D space.

### simlib (`src/simlib`)

- **kernels.py:** Functions for particle interactions (P2P with gradients, jerks, and snaps) and multipole translations using spherical harmonics (p2m, m2m, m2l, and l2l).
- **simulators.py:** Defines the n-body Simulator interface, and classes `nbodyDirectSimulator` and `fmmSimulator`.
- **acceptanceCriterion.py:** Contains acceptance criteria for multipole approximations.

### utils (`src/utils`)

- **performanceTest.py:** Benchmarks and plots the performance of direct summation versus FMM as the particle count increases.
- **visualization.py:** Leverages VTK to visualize and animate point cloud data.


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
├── traceData # Contains all the traceData
│   └── trace1.npy
│   └── ...
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

## References
- W. Dehnen, "A fast multipole method for stellar dynamics," Comput. Astrophys., vol. 1, no. 1, 2014, [doi:10.1186/s40668-014-0001-7](https://link.springer.com/article/10.1186/s40668-014-0001-7).
  
- K. Taura, J. Nakashima, R. Yokota, and N. Maruyama, "A task parallel implementation of fast multipole methods," in 2012 SC Companion: High Performance Computing, Networking, Storage and Analysis, Salt Lake City, UT, USA, 2012, pp. 617–625, [doi:10.1109/SC.Companion.2012.86](https://ieeexplore.ieee.org/document/6495868).