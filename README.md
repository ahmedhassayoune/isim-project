# Adaptive Quad Mesh Simplification

This repository contains Python scripts and tools for adaptive quad mesh simplification based on the techniques described in the paper "Adaptive Quad Mesh Simplification" by Bozzi et al. (2010).

## Introduction
Quad meshes are widely used in computer graphics and simulation for representing surfaces. Mesh simplification techniques aim to reduce the complexity of quad meshes while preserving important geometric and topological features. This project implements adaptive quad mesh simplification algorithms to achieve efficient and high-quality mesh reduction.

## Simplification Protocol
The mesh simplification process follows a specific protocol involving several steps:
1. **Triangulation**: Convert the classic mesh into a triangulated mesh.
2. **Tri-to-Quad**: Apply the Tri-to-Quad algorithm to convert the triangulated mesh into a quad mesh.
3. **Simplification**: Use adaptive quad mesh simplification techniques to reduce the complexity of the quad mesh.

## Dependencies
Blender Python API (blender-python-api=4.0.0): The project leverages Blender's Python API for mesh operations and visualization.

## References
- Bozzi, E., Panozzo, D., Puppo, E., and Tarini, M. (2010). "Adaptive Quad Mesh Simplification." Eurographics Italian Chapter Conference, pp. 25-30. [PDF](https://cims.nyu.edu/gcl/papers/EGIT10-BozPanPupetall.pdf)
- Tarini, M., Pietroni, N., Cignoni, P., Panozzo, D., Puppo, E. "Practical Quad Simplification." Eurographics 2010 [PDF](https://vcg.isti.cnr.it/quadSemplif/Tarini%20Pietroni%20Cignoni%20Panozzo%20Puppo%20-%20Practical%20Quad%20Semplification%20-%20EG%202010.pdf)