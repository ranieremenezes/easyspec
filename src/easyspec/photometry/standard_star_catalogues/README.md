# Catalogues of photometric standard stars

csv files read by `easyspec.photometry` (`photometry.standard_star_catalogue()`, `fit_standard_star_solution(catalogue=...)`).

| File | Catalogue | Stars | CDS |
|---|---|---|---|
| `landolt_1992.csv` | Landolt (1992), AJ 104, 340 | UBVRI, around the celestial equator | II/183A |
| `landolt_2009.csv` | Landolt (2009), AJ 137, 4186 | UBVRI, update and extension of 1992, with proper motions | J/AJ/137/4186 |
| `landolt_2013.csv` | Landolt (2013), AJ 146, 131 | UBVRI, declination +50 deg | J/AJ/146/131 |
| `clem_landolt_2013.csv` | Clem & Landolt (2013), AJ 146, 88 | faint UBVRI (V to ~22) in the Landolt fields | J/AJ/146/88 |
| `clem_landolt_2016.csv` | Clem & Landolt (2016), AJ 152, 91 | faint UBVRI at declination +50 deg | J/AJ/152/91 |

Names accepted by easyspec: the file names, plus the aliases `"landolt"` (2009 + 2013), `"clem_landolt"` (2013 + 2016).

Columns: `name`, `field`, `ra`, `dec` (deg, J2000), magnitudes and colors as in the catalogue (e.g. `V`, `B-V`, `U-B`, `V-R`, `R-I`, `V-I`;
the individual bands are derived from them when the catalogue is read), their errors (`e_V`, `e_B-V`, ...) and the number of observations.

The files are created by `build_standard_star_catalogues.py` (run it once with an internet connection). Any other catalogue (e.g. the
standard fields of Stetson, from CADC/CANFAR, or the SDSS u'g'r'i'z' standards of Smith et al. 2002, which are not in VizieR) can be used as a csv file with the same columns.
