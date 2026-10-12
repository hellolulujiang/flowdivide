"""data_06_basin_table_extracts: the two extracts of the basin table that Figures 6 and A4 print.
Reads the published global/table/basin_table_fine_south-america.csv and writes into data/:
    basin_table_fine_south-america_head.txt    the header, the first five rows and the last row (Figure 6b)
    example_basin_10827_row.txt                the header and the row of basin 10827, the example basin (Figures 6b, A4b)

    python3 data_06_basin_table_extracts.py
"""
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

CONTINENT = "south-america"
EXAMPLE_BASIN = 10827
TABLE = os.path.join(paths.HYDROSHEDS30M_ROOT, CONTINENT, "global", "table", "basin_table_fine_%s.csv" % CONTINENT)


def main():
    out = paths.DATA_DIRECTORY
    wanted = str(EXAMPLE_BASIN)
    with open(TABLE) as handle:
        header = handle.readline()
        first_five = [handle.readline() for _ in range(5)]
        example_row = next((line for line in first_five if line.split(None, 1)[0] == wanted), None)
        last = collections.deque(first_five, maxlen=1)
        for line in handle:
            if example_row is None and line.split(None, 1)[0] == wanted:
                example_row = line
            last.append(line)
    if example_row is None:
        raise SystemExit("basin %d is not in %s" % (EXAMPLE_BASIN, TABLE))
    with open(os.path.join(out, "basin_table_fine_%s_head.txt" % CONTINENT), "w") as handle:
        handle.writelines([header] + first_five + list(last))
    with open(os.path.join(out, "example_basin_%d_row.txt" % EXAMPLE_BASIN), "w") as handle:
        handle.writelines([header, example_row])
    print("head: rows 1-5 and basin %s; example: basin %d" % (last[0].split(None, 1)[0], EXAMPLE_BASIN))

if __name__ == "__main__":
    main()
