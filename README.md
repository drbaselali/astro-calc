# astro-calc

This repository demonstrates the application of statistics and machine learning on astrophysical systems for purposes of orbital and dynamical inference.

The current project explores the following methods: 

* Bilby- MCMC (https://arxiv.org/abs/2106.08730)
* Ultranest (https://johannesbuchner.github.io/UltraNest/index.html)
* HDBSCAN (https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)

The first two approaches are used to infer the orbital elements for objects that lack radial velocity data, while the third is employed to find any underlying structure in the orbital parameter space.

**Key Findings**

**Bilby-MCMC**


![Simulation result](figures/bilby.png)

According to the figure, the method fails to recover the second modes of both of the argument of the pericenter and the longitude of ascending node, which are expected when radial velocity data are missing.

**Ultranest**

![Simulation result](figures/S20.png)

As demonstrated above nested sampling methods perform significantly better than the traditional MCMC methods. 

**HDBSCAN**

![Simulation result](figures/hdbscan.png)

After inferring the orbital solutions, HDBSCAN is applied to the specific angular momentum vectors in three-dimensional space to identify structure within the derived orbital population.

Author

Dr. Basel Ali
