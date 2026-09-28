# Notice on data

The MIT terms in [`LICENSE`](LICENSE) cover the code and the documentation in this
repository. The repository holds no data: the tests either make small grids of their own or read a case
directory cut from a run of the package on data you provide. The three figures in `docs/media/` are
figures of the paper that illustrate the documentation and fall under its terms: `workflow.png` is a
diagram, and `amazon_cut_into_regions.png` and `basin_groups_south_america.png` are maps drawn from HydroSHEDS v2
and HydroBASINS (credited below). They are pictures, not data products.

FlowDivide runs on flow-direction grids and basin polygons from other producers, which you download
yourself and use under their terms:

* **MERIT Hydro** (Yamazaki et al., 2019, <https://doi.org/10.1029/2019WR024873>), which its authors
  distribute under CC BY-NC 4.0 or ODbL 1.0: <https://global-hydrodynamics.github.io/MERIT_Hydro/>
* **HydroSHEDS v2** and **HydroBASINS** (Lehner and Grill, 2013, <https://doi.org/10.1002/hyp.9740>),
  under the terms given at <https://www.hydrosheds.org>

What you make with FlowDivide from these grids is derived from them; read the producer's terms before
you distribute it.
