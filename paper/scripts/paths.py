"""Where the scripts of the FlowDivide paper read and write.

The figure scripts (fig*.py) read only the data folder and write the figures folder.  The data scripts
(data*.py) build the data folder again from inputs that are not in this repository; give their locations in
environment variables:

    FLOWDIVIDE_PRODUCTS    the products of FlowDivide on HydroSHEDS v2.0 (one folder per continent:
                           south-america/, north-america/), as published at https://fullhydro.org
    FLOWDIVIDE_BENCHMARK   the timing records of the benchmark of Fig. 7
    FLOWDIVIDE_RUNS        the logs of the package runs of Fig. 8
    HYDROSHEDS_V2          HydroSHEDS v2.0 as distributed (https://www.hydrosheds.org)

data_08_package_run.py also asks the package for the sizes of the three grids, which it reads from its own
variables: FLOWDIVIDE_HYDROSHEDS (the same directory as HYDROSHEDS_V2) and FLOWDIVIDE_MERIT (MERIT Hydro's flow
directions, one file).

FLOWDIVIDE_PAPER_DATA moves the data folder, FLOWDIVIDE_PACKAGE the flowdivide package (by default the root of
this repository), and FLOWDIVIDE_PALETTE the file of the eight colours of the views.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
DATA_DIRECTORY = os.environ.get("FLOWDIVIDE_PAPER_DATA", os.path.join(PAPER, "data"))
FIGURE_DIRECTORY = os.path.join(PAPER, "figures")
PACKAGE_DIRECTORY = os.environ.get("FLOWDIVIDE_PACKAGE", os.path.dirname(PAPER))
COLOR_PALETTE = os.environ.get("FLOWDIVIDE_PALETTE", os.path.join(DATA_DIRECTORY, "color_id_palette.csv"))

# the inputs of the data scripts, set only when their environment variable is set
INPUTS = {"HYDROSHEDS30M_ROOT": "FLOWDIVIDE_PRODUCTS", "BENCHMARK_RUNS": "FLOWDIVIDE_BENCHMARK",
          "PACKAGE_RUNS": "FLOWDIVIDE_RUNS", "RAW_HYDROSHEDS": "HYDROSHEDS_V2"}
for _name, _variable in INPUTS.items():
    if os.environ.get(_variable):
        globals()[_name] = os.environ[_variable]


def __getattr__(name):
    if name in INPUTS:
        raise SystemExit("this script reads %s: set the environment variable %s (see paths.py)" % (name, INPUTS[name]))
    raise AttributeError(name)
