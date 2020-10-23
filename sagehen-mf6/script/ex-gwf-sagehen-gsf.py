# ## LGR example with MVR for SFR between two models
#
# This script reproduces the model in Mehl and Hill (2013).

# ### MODFLOW 6 LGR Problem Setup

# Append to system path to include the common subdirectory

import os
import sys

sys.path.append(os.path.join("..", "common"))

# Imports

import flopy
import numpy as np
import config
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf
from figspecs import USGSFigure

sys.path.append(os.path.join("..", "data", "sagehen-gsf"))
import build_sagehen_helper_funcs as sageBld

mf6exe = os.path.abspath(config.mf6_exe)
assert os.path.isfile(mf6exe)
print(mf6exe)
exe_name_mf = config.mf2005_exe
exe_name_mt = config.mt3dms_exe

# Set figure properties specific to this problem

figure_size = (7, 5)

# Base simulation and model name and workspace

ws = config.base_ws
example_name = "ex-gwf-sagehen-gsf"

# Model units

length_units = "meters"
time_units = "days"

# Table

nlay = 2  # Number of layers in parent model
nrow = 73  # Number of rows in parent model
ncol = 81  # Number of columns in parent model
delr = 90.  # Parent model column width ($m$)
delc = 90.  # Parent model row width ($m$)
k11 = "varies"  # Horizontal hydraulic conductivity ($m/d$)

# Additional model input preparation
# Time related variables
perlen = [1, 5844]
nper = len(perlen)
nstp = [1, 5844]
tsmult = [1.0, 1.0]

# Further parent model grid discretization

# from mf-nwt .dis file
dat_pth = os.path.join("..","data","sagehen-gsf")
top = np.loadtxt(os.path.join(dat_pth,"orig_dis_input","top1.txt"))
bot1 = np.loadtxt(os.path.join(dat_pth,"orig_dis_input","bot1.txt"))
bot2 = np.loadtxt(os.path.join(dat_pth,"orig_dis_input","bot2.txt"))
botm = [bot1, bot2]
# from mf-nwt .bas file
idomain1 = np.loadtxt(os.path.join(dat_pth,"orig_bas_input","ibnd1.txt"))
idomain2 = np.loadtxt(os.path.join(dat_pth,"orig_bas_input","ibnd2.txt"))
strt1 = np.loadtxt(os.path.join(dat_pth,"orig_bas_input","strt1.txt"))
strt2 = np.loadtxt(os.path.join(dat_pth,"orig_bas_input","strt2.txt"))
# peel out locations of negative values for setting constant head data
tmp1 = np.where(idomain1 < 0)
listOfChdCoords1 = list(zip(np.zeros_like(tmp1[0]), tmp1[0], tmp1[1]))
# get the corresponding constant head values
if(len(listOfChdCoords1) > 0):
    chd_lay1 = list(np.take(strt1 , np.ravel_multi_index(tmp1, strt1.shape)))
# work on layer 2
tmp2 = np.where(idomain2 < 0)
listOfChdCoords2 = list(zip(np.ones_like(tmp2[0]), tmp2[0], tmp2[1]))
if(len(listOfChdCoords2) > 0):
    chd_lay2 = list(np.take(strt2 , np.ravel_multi_index(tmp2, strt2.shape)))
# Get the constant head data into a flopy-compatible format
listOfChdCoords = listOfChdCoords1 + listOfChdCoords2
chd_vals = chd_lay1 + chd_lay2
chdspd = []
for i in np.arange(len(listOfChdCoords)):
    chdspd.append([listOfChdCoords[i], chd_vals[i]])
# finally, get rid of the negative values in idomain since mf6 treats negatives like zeros
idomain = [np.abs(idomain1), np.abs(idomain2)]

# from mf-nwt .upw file
k11_lay1 = np.loadtxt(os.path.join(dat_pth,"orig_upw_input","hk1.txt"))
k11_lay2 = np.loadtxt(os.path.join(dat_pth,"orig_upw_input","hk2.txt"))
k11 = [k11_lay1, k11_lay2]
sy_lay1 = np.loadtxt(os.path.join(dat_pth,"orig_upw_input","sy1.txt"))
sy = [sy_lay1, sy_lay1]
# sy_lay2 not in original problem (laytyp = 0 in layer 2)
k33_lay1 = np.loadtxt(os.path.join(dat_pth,"orig_upw_input","vk1.txt"))
k33_lay2 = np.loadtxt(os.path.join(dat_pth,"orig_upw_input","vk2.txt"))
k33 = [k33_lay1, k33_lay2]

icelltype = [1, 0]  # Water table resides in layer 1
iconvert = [np.ones_like(strt1), np.zeros_like(strt2)]

# Solver settings

nouter, ninner = 300, 500
hclose, rclose, relax = 1e-3, 1e-2, 0.97

# #### Prepping input for SFR package 
# Package_data information

# Define the connections
conns = sageBld.gen_mf6_sfr_connections()

# These are zero based
sfrcells = [
    (0, 38, 14),
    (0, 38, 15),
    (0, 38, 16),
    (0, 37, 16),
    (0, 37, 17),
    (0, 37, 18),
    (0, 37, 19),
    (0, 37, 20),
    (0, 37, 21),
    (0, 37, 22),
    (0, 36, 23),
    (0, 36, 24),
    (0, 35, 24),
    (0, 35, 25),
    (0, 34, 25),
    (0, 34, 26),
    (0, 35, 27),
    (0, 35, 28),
    (0, 35, 29),
    (0, 35, 30),
    (0, 35, 31),
    (0, 34, 31),
    (0, 34, 32),
    (0, 33, 32),
    (0, 33, 33),
    (0, 32, 33),
    (0, 32, 34),
    (0, 31, 34),
    (0, 31, 35),
    (0, 31, 36),
    (0, 52, 36),
    (0, 51, 36),
    (0, 51, 37),
    (0, 50, 37),
    (0, 49, 37),
    (0, 49, 38),
    (0, 48, 38),
    (0, 47, 38),
    (0, 46, 38),
    (0, 46, 39),
    (0, 45, 39),
    (0, 44, 39),
    (0, 44, 38),
    (0, 43, 38),
    (0, 42, 37),
    (0, 41, 37),
    (0, 40, 37),
    (0, 39, 37),
    (0, 39, 38),
    (0, 38, 38),
    (0, 37, 38),
    (0, 36, 39),
    (0, 35, 39),
    (0, 35, 40),
    (0, 34, 40),
    (0, 33, 40),
    (0, 32, 40),
    (0, 31, 40),
    (0, 30, 41),
    (0, 30, 32),
    (0, 30, 33),
    (0, 30, 34),
    (0, 30, 35),
    (0, 30, 36),
    (0, 47, 47),
    (0, 46, 47),
    (0, 45, 47),
    (0, 45, 46),
    (0, 44, 46),
    (0, 43, 46),
    (0, 42, 46),
    (0, 41, 46),
    (0, 40, 46),
    (0, 40, 47),
    (0, 39, 47),
    (0, 38, 47),
    (0, 37, 46),
    (0, 36, 46),
    (0, 35, 47),
    (0, 34, 47),
    (0, 34, 48),
    (0, 33, 48),
    (0, 33, 49),
    (0, 32, 49),
    (0, 54, 71),
    (0, 53, 71),
    (0, 52, 71),
    (0, 51, 71),
    (0, 50, 71),
    (0, 49, 72),
    (0, 48, 72),
    (0, 47, 72),
    (0, 47, 73),
    (0, 46, 73),
    (0, 45, 74),
    (0, 44, 74),
    (0, 44, 75),
    (0, 43, 75),
    (0, 44, 61),
    (0, 43, 61),
    (0, 42, 61),
    (0, 42, 62),
    (0, 41, 62),
    (0, 40, 62),
    (0, 39, 62),
    (0, 23, 54),
    (0, 24, 54),
    (0, 24, 55),
    (0, 25, 55),
    (0, 26, 55),
    (0, 27, 56),
    (0, 28, 56),
    (0, 29, 56),
    (0, 30, 56),
    (0, 31, 56),
    (0, 32, 56),
    (0, 32, 57),
    (0, 33, 57),
    (0, 33, 58),
    (0, 34, 58),
    (0, 35, 58),
    (0, 36, 59),
    (0, 22, 70),
    (0, 23, 70),
    (0, 24, 70),
    (0, 25, 70),
    (0, 26, 71),
    (0, 26, 72),
    (0, 27, 72),
    (0, 28, 72),
    (0, 29, 72),
    (0, 30, 72),
    (0, 31, 72),
    (0, 32, 72),
    (0, 33, 72),
    (0, 33, 73),
    (0, 34, 73),
    (0, 35, 73),
    (0, 35, 72),
    (0, 36, 72),
    (0, 37, 71),
    (0, 38, 71),
    (0, 39, 71),
    (0, 40, 71),
    (0, 41, 71),
    (0, 41, 72),
    (0, 30, 37),
    (0, 30, 38),
    (0, 30, 39),
    (0, 30, 40),
    (0, 30, 41),
    (0, 29, 41),
    (0, 29, 42),
    (0, 29, 43),
    (0, 28, 43),
    (0, 28, 44),
    (0, 28, 45),
    (0, 28, 46),
    (0, 29, 46),
    (0, 29, 47),
    (0, 30, 48),
    (0, 31, 49),
    (0, 31, 50),
    (0, 32, 51),
    (0, 32, 52),
    (0, 33, 52),
    (0, 33, 53),
    (0, 34, 53),
    (0, 34, 54),
    (0, 34, 55),
    (0, 35, 56),
    (0, 35, 57),
    (0, 35, 58),
    (0, 36, 58),
    (0, 36, 59),
    (0, 37, 59),
    (0, 37, 60),
    (0, 37, 61),
    (0, 37, 62),
    (0, 38, 62),
    (0, 38, 63),
    (0, 38, 64),
    (0, 39, 64),
    (0, 39, 65),
    (0, 39, 66),
    (0, 39, 67),
    (0, 40, 68),
    (0, 40, 69),
    (0, 41, 70),
    (0, 41, 71),
    (0, 41, 72),
    (0, 41, 72),
    (0, 42, 72),
    (0, 42, 73),
    (0, 42, 74),
    (0, 43, 74),
    (0, 43, 75),
    (0, 43, 76),
    (0, 43, 77),
    (0, 43, 78),
    (0, 44, 78)
]

rlen = [
    102.0,
     90.0,
     60.0,
     30.0,
    102.0,
     90.0,
     90.0,
     90.0,
    102.0,
    102.0,
    102.0,
     72.0,
     30.0,
     72.0,
     30.0,
     90.0,
    102.0,
     90.0,
     90.0,
    102.0,
     30.0,
     72.0,
     30.0,
     72.0,
     30.0,
     60.0,
     72.0,
     30.0,
    102.0,
     90.0,
     90.0,
     72.0,
     30.0,
    102.0,
     60.0,
     30.0,
     90.0,
    102.0,
     60.0,
     30.0,
     90.0,
     30.0,
     60.0,
    102.0,
    102.0,
     90.0,
    102.0,
     30.0,
     60.0,
    102.0,
    102.0,
    102.0,
     60.0,
     30.0,
    102.0,
     90.0,
     90.0,
    102.0,
    114.0,
     90.0,
     90.0,
    102.0,
    102.0,
     90.0,
     90.0,
    102.0,
     30.0,
     60.0,
     90.0,
    102.0,
     90.0,
     90.0,
     60.0,
     30.0,
     90.0,
     90.0,
     90.0,
     90.0,
    102.0,
     30.0,
     60.0,
     72.0,
     30.0,
    114.0,
     90.0,
     90.0,
     90.0,
    102.0,
    102.0,
    102.0,
    102.0,
     30.0,
     60.0,
    102.0,
    102.0,
     30.0,
     72.0,
     30.0,
     90.0,
    102.0,
     30.0,
     60.0,
    102.0,
     90.0,
    102.0,
     60.0,
     30.0,
     60.0,
    102.0,
     90.0,
     90.0,
     90.0,
     90.0,
     90.0,
    102.0,
     72.0,
     30.0,
     72.0,
     30.0,
    102.0,
     90.0,
    114.0,
     90.0,
    102.0,
     90.0,
    102.0,
    102.0,
     30.0,
    102.0,
     90.0,
     90.0,
     90.0,
     90.0,
     90.0,
     30.0,
     60.0,
     90.0,
     30.0,
     60.0,
    102.0,
     90.0,
     90.0,
     90.0,
     90.0,
     30.0,
     30.0,
     90.0,
     90.0,
     90.0,
     90.0,
     72.0,
     30.0,
     90.0,
     60.0,
     30.0,
     90.0,
     90.0,
     30.0,
     60.0,
    102.0,
    114.0,
    114.0,
     90.0,
    102.0,
     30.0,
     72.0,
     60.0,
     30.0,
    102.0,
    102.0,
    114.0,
     90.0,
     60.0,
     30.0,
     72.0,
     30.0,
    102.0,
    102.0,
     30.0,
     60.0,
    120.0,
     60.0,
     30.0,
     90.0,
     90.0,
     90.0,
     90.0,
    114.0,
    102.0,
     90.0,
     30.0,
     30.0,
     30.0,
    102.0,
     60.0,
     30.0,
     90.0,
     90.0,
     90.0,
     60.0,
     30.0
]

rwid = 3.0
rgrd = [
    0.042,
    0.064,
    0.083,
    0.081,
    0.062,
    0.065,
    0.089,
    0.097,
    0.141,
    0.186,
    0.217,
    0.203,
    0.206,
    0.206,
    0.144,
    0.147,
    0.135,
    0.129,
    0.124,
    0.136,
    0.162,
    0.147,
    0.157,
    0.147,
    0.115,
    0.117,
    0.111,
    0.120,
    0.099,
    0.073,
    0.037,
    0.038,
    0.060,
    0.048,
    0.024,
    0.029,
    0.032,
    0.028,
    0.024,
    0.029,
    0.033,
    0.038,
    0.032,
    0.038,
    0.051,
    0.047,
    0.037,
    0.063,
    0.063,
    0.049,
    0.069,
    0.077,
    0.063,
    0.045,
    0.037,
    0.043,
    0.048,
    0.054,
    0.065,
    0.067,
    0.091,
    0.091,
    0.071,
    0.073,
    0.021,
    0.031,
    0.045,
    0.033,
    0.029,
    0.042,
    0.075,
    0.103,
    0.092,
    0.095,
    0.087,
    0.083,
    0.094,
    0.102,
    0.093,
    0.081,
    0.099,
    0.077,
    0.057,
    0.056,
    0.044,
    0.050,
    0.075,
    0.076,
    0.074,
    0.074,
    0.071,
    0.072,
    0.056,
    0.060,
    0.048,
    0.043,
    0.049,
    0.039,
    0.042,
    0.056,
    0.081,
    0.071,
    0.068,
    0.068,
    0.068,
    0.044,
    0.078,
    0.071,
    0.051,
    0.054,
    0.056,
    0.056,
    0.050,
    0.048,
    0.038,
    0.022,
    0.049,
    0.059,
    0.043,
    0.043,
    0.045,
    0.049,
    0.042,
    0.031,
    0.016,
    0.010,
    0.012,
    0.015,
    0.012,
    0.011,
    0.022,
    0.044,
    0.056,
    0.060,
    0.114,
    0.100,
    0.067,
    0.086,
    0.127,
    0.141,
    0.118,
    0.100,
    0.083,
    0.087,
    0.100,
    0.067,
    0.056,
    0.083,
    0.100,
    0.076,
    0.045,
    0.020,
    0.053,
    0.042,
    0.038,
    0.047,
    0.047,
    0.057,
    0.040,
    0.032,
    0.045,
    0.053,
    0.042,
    0.049,
    0.094,
    0.085,
    0.036,
    0.027,
    0.030,
    0.033,
    0.024,
    0.017,
    0.025,
    0.021,
    0.015,
    0.010,
    0.010,
    0.012,
    0.018,
    0.022,
    0.017,
    0.019,
    0.010,
    0.013,
    0.022,
    0.017,
    0.021,
    0.043,
    0.044,
    0.038,
    0.050,
    0.033,
    0.021,
    0.020,
    0.024,
    0.029,
    0.020,
    0.011,
    0.024,
    0.033,
    0.022
]

rtp = [
    2356.5,
    2352.5,
    2345.5,
    2342.5,
    2336.5,
    2332.5,
    2324.5,
    2316.5,
    2306.5,
    2288.5,
    2268.5,
    2247.5,
    2240.5,
    2226.5,
    2219.5,
    2210.5,
    2196.5,
    2184.5,
    2172.5,
    2161.5,
    2150.5,
    2142.5,
    2135.5,
    2126.5,
    2120.5,
    2115.5,
    2107.5,
    2102.5,
    2093.5,
    2086.5,
    2147.5,
    2144.5,
    2142.5,
    2137.5,
    2135.5,
    2134.5,
    2132.5,
    2129.5,
    2127.5,
    2126.5,
    2124.5,
    2122.5,
    2120.5,
    2118.5,
    2113.5,
    2108.5,
    2104.5,
    2102.5,
    2097.5,
    2094.5,
    2088.5,
    2080.5,
    2074.5,
    2072.5,
    2069.5,
    2066.5,
    2061.5,
    2057.5,
    2049.5,
    2116.5,
    2110.5,
    2099.5,
    2092.5,
    2085.5,
    2113.5,
    2111.5,
    2109.5,
    2107.5,
    2105.5,
    2102.5,
    2097.5,
    2088.5,
    2080.5,
    2077.5,
    2070.5,
    2064.5,
    2055.5,
    2047.5,
    2036.5,
    2032.5,
    2027.5,
    2021.5,
    2018.5,
    2014.5,
    2001.5,
    1997.5,
    1992.5,
    1983.5,
    1977.5,
    1968.5,
    1962.5,
    1956.5,
    1954.5,
    1949.5,
    1943.5,
    1941.5,
    1938.5,
    1936.5,
    2004.5,
    1999.5,
    1995.5,
    1991.5,
    1985.5,
    1979.5,
    1973.5,
    2033.5,
    2031.5,
    2026.5,
    2022.5,
    2017.5,
    2012.5,
    2007.5,
    2002.5,
    1998.5,
    1993.5,
    1991.5,
    1990.5,
    1986.5,
    1984.5,
    1981.5,
    1977.5,
    1972.5,
    2052.5,
    2048.5,
    2046.5,
    2045.5,
    2044.5,
    2043.5,
    2042.5,
    2041.5,
    2040.5,
    2037.5,
    2032.5,
    2027.5,
    2023.5,
    2015.5,
    2011.5,
    2006.5,
    2002.5,
    1990.5,
    1977.5,
    1968.5,
    1959.5,
    1953.5,
    1946.5,
    1944.5,
    2077.5,
    2072.5,
    2062.5,
    2054.5,
    2049.5,
    2048.5,
    2042.5,
    2038.5,
    2037.5,
    2034.5,
    2030.5,
    2027.5,
    2024.5,
    2022.5,
    2018.5,
    2012.5,
    2007.5,
    2003.5,
    1999.5,
    1992.5,
    1989.5,
    1988.5,
    1986.5,
    1983.5,
    1979.5,
    1978.5,
    1976.5,
    1975.5,
    1974.5,
    1973.5,
    1971.5,
    1970.5,
    1969.5,
    1968.5,
    1967.5,
    1966.5,
    1965.5,
    1964.5,
    1962.5,
    1960.5,
    1959.5,
    1956.5,
    1950.5,
    1947.5,
    1944.5,
    1943.5,
    1942.5,
    1941.5,
    1939.5,
    1938.5,
    1936.5,
    1935.5,
    1934.5,
    1931.5,
    1930.5
]

rbth = 1.0
rhk = 5.0
man = 0.04
ustrf = 1.0
ndv = 0
pkdat = []
for i in np.arange(len(rlen)):
    ncon = len(conns[i]) - 1
    pkdat.append(
        (
            i,
            sfrcells[i],
            rlen[i],
            rwid,
            rgrd[i],
            rtp[i],
            rbth,
            rhk,
            man,
            ncon,
            ustrf,
            ndv,
        )
    )

# #### Prepping input for UZF package 
# Package_data information

iuzbnd = np.loadtxt(os.path.join(dat_pth,"orig_uzf_input","iuzbnd.txt"))
thts = np.loadtxt(os.path.join(dat_pth,"orig_uzf_input","thts.txt"))
uzk33 = np.loadtxt(os.path.join(dat_pth,"orig_uzf_input","uz_vk_cln.txt"))
finf = np.loadtxt(os.path.join(dat_pth,"orig_uzf_input","finf.txt"))

pet_ss = 0.008    # mf6io.pdf: Must always be specified, even when not used
extdp_ss = 1.0    # mf6io.pdf: Must always be specified, even when not used
extwc_ss = 0.055  # mf6io.pdf: Must always be specified, even when not used
ha = 0.
hroot = 0.
rootact = 0.

extdp = extdp_ss
extwc = extwc_ss

uzf_packagedata = []
pd0             = []
iuzno_cell_dict = {}
iuzno_dict_rev  = {}
iuzno           = 0
surfdep         = 0.5
# Set up the UZF static variables
nuzfcells = 0
for k in range(nlay):
    for i in range(0, iuzbnd.shape[0] - 1):
        for j in range(0,iuzbnd.shape[1] - 1):
            if iuzbnd[i, j] != 0:
                nuzfcells += 1
                if k == 0:
                    lflag = 1
                    iuzno_cell_dict.update({(i, j): iuzno})  # establish new dictionary entry for current cell 
                                                             # addresses & iuzno connections are both 0-based
                    iuzno_dict_rev.update({iuzno: (i, j)})   # For post-processing the mvr output, need a dict with iuzno as key
                else:
                    lflag = 0
                
                # Set the vertical connection, which is the cell below
                # For now, using only the GSFLOW version of Sagehen, only the first layer hosts UZF objects
                ivertcon = -1
                #ivertcon =  iuzno + int(iuzfbnd.sum())
                #if k == nlay - 1: ivertcon = -1       # adjust if on bottom layer (no underlying conn.)
                #                                      # Keep in mind 0-based adjustment (so ivertcon==-1 -> 0)
                
                surfdep = 1.0
                vks = uzk33[i, j]
                thtr = 0.01
                thtsx = thts[i, j]
                thti = 0.08
                eps = 4.0
                
                # Set the boundname for the land surface cells
                bndnm = 'sageSurf'
                
                # <iuzno> <cellid(ncelldim)> <landflag> <ivertcon> <surfdep> <vks> <thtr> <thts> <thti> <eps> [<boundname>]
                uz = [iuzno,      (k, i, j),     lflag,  ivertcon,  surfdep,  vks,  thtr,  thtsx,  thti,  eps,   bndnm]
                uzf_packagedata.append(uz)
            
                # steady-state values can be set here
                if lflag:
                    finf_ss = finf[i, j]
                    pd0.append((iuzno, finf_ss, pet_ss, extdp_ss, extwc_ss, ha, hroot, rootact))
                
                iuzno += 1

# Store the steady state uzf stresses in dictionary
uzf_perioddata = {0: pd0}


# ### Function to build models
#
# MODFLOW 6 flopy simulation object (sim) is returned if building the model

def build_model(sim_name, silent=False):
    if config.buildModel:

        # Instantiate the MODFLOW 6 simulation
        name = "sagehen-gsf"
        gwfname = "gwf_" + name
        sim_ws = os.path.join(ws, sim_name)
        sim = flopy.mf6.MFSimulation(
            sim_name=sim_name,
            version="mf6",
            sim_ws=sim_ws,
            exe_name=mf6exe,
            continue_=True,
        )

        # Instantiating MODFLOW 6 time discretization
        tdis_rc = []
        for i in range(len(perlen)):
            tdis_rc.append((perlen[i], nstp[i], tsmult[i]))
        flopy.mf6.ModflowTdis(
            sim, nper=nper, perioddata=tdis_rc, time_units=time_units
        )

        # Instantiating MODFLOW 6 groundwater flow model
        gwfname = gwfname
        gwf = flopy.mf6.ModflowGwf(
            sim,
            modelname=gwfname,
            save_flows=True,
            newtonoptions=True,
            model_nam_file="{}.nam".format(gwfname),
        )

        # Instantiating MODFLOW 6 solver for flow model
        imsgwf = flopy.mf6.ModflowIms(
            sim,
            print_option="summary",
            complexity="complex",
            outer_dvclose=hclose,
            outer_maximum=nouter,
            under_relaxation="dbd",
            linear_acceleration="BICGSTAB",
            under_relaxation_theta=0.7,
            under_relaxation_kappa=0.08,
            under_relaxation_gamma=0.05,
            under_relaxation_momentum=0.0,
            backtracking_number=20,
            backtracking_tolerance=2.0,
            backtracking_reduction_factor=0.2,
            backtracking_residual_limit=5.0e-4,
            inner_dvclose=hclose,
            rcloserecord=[0.0001, "relative_rclose"],
            inner_maximum=ninner,
            relaxation_factor=relax,
            number_orthogonalizations=2,
            preconditioner_levels=8,
            preconditioner_drop_tolerance=0.001,
            filename="{}.ims".format(gwfname)
        )
        sim.register_ims_package(imsgwf, [gwf.name])

        # Instantiating MODFLOW 6 discretization package
        flopy.mf6.ModflowGwfdis(
            gwf,
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            delr=delr,
            delc=delc,
            top=top,
            botm=botm,
            idomain=idomain,
            filename="{}.dis".format(gwfname)
        )

        # Instantiating MODFLOW 6 initial conditions package for flow model
        strt = [strt1, strt2]
        flopy.mf6.ModflowGwfic(
            gwf, 
            strt=strt, 
            filename="{}.ic".format(gwfname)
        )

        # Instantiating MODFLOW 6 node-property flow package
        flopy.mf6.ModflowGwfnpf(
            gwf,
            save_flows=False,
            alternative_cell_averaging="AMT-HMK",
            icelltype=icelltype,
            k=k11,
            k33=k33,
            save_specific_discharge=False,
            filename="{}.npf".format(gwfname)
        )

        # Instantiate MODFLOW 6 storage package 
        flopy.mf6.ModflowGwfsto(
            gwf, 
            ss=2e-6, 
            sy=sy,
            iconvert=iconvert,
            steady_state={0:True},
            transient={1:True},
            filename='{}.sto'.format(gwfname)
        )
        
        # Instantiating MODFLOW 6 output control package for flow model
        flopy.mf6.ModflowGwfoc(
            gwf,
            budget_filerecord="{}.bud".format(gwfname),
            head_filerecord="{}.hds".format(gwfname),
            headprintrecord=[
                ("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")
            ],
            saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
            printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
        )

        # Instantiating MODFLOW 6 constant head package
        chdspdx = {0: chdspd}
        flopy.mf6.ModflowGwfchd(
            gwf,
            maxbound=len(chdspd),
            stress_period_data=chdspdx,
            save_flows=False,
            pname="CHD-1",
            filename="{}.chd".format(gwfname),
        )
        
        # Instantiating MODFLOW 6 streamflow routing package
        flopy.mf6.ModflowGwfsfr(
            gwf,
            print_stage=False,
            print_flows=False,
            budget_filerecord=gwfname + ".sfr.bud",
            save_flows=True,
            mover=False,
            pname="SFR-1",
            unit_conversion=86400.0,
            boundnames=True,
            nreaches=len(conns),
            packagedata=pkdat,
            connectiondata=conns,
            perioddata=None,
            filename="{}.sfr".format(gwfname),
        )
        
        # Instantiating MODFLOW 6 unsaturated zone flow package
        flopy.mf6.ModflowGwfuzf(
            gwf, 
            nuzfcells=nuzfcells, 
            boundnames=True,
            ntrailwaves=15, 
            nwavesets=150, 
            print_flows=False,
            save_flows=True,
            simulate_et=False, 
            packagedata=uzf_packagedata, 
            perioddata=uzf_perioddata,
            budget_filerecord='{}.uzf.bud'.format(gwfname),
            pname='UZF-1',
            filename='{}.uzf'.format(gwfname)
        )
        
        return sim
    return None

# Function to write model files

def write_model(sim, silent=True):
    if config.writeModel:
        sim.write_simulation(silent=silent)

# Function to run the model. True is returned if the model runs successfully

def run_model(sim, silent=True):
    success = True
    if config.runModel:
        success = False
        success, buff = sim.run_simulation(silent=silent)
        if not success:
            print(buff)
    return success

# Function to plot the model results

def plot_results(mf6, idx):
    if config.plotModel:
        print("Plotting model results...")
        sim_name = mf6.name
        fs = USGSFigure(figure_type="graph", verbose=False)
        
        # Generate a plot of FINF distribution
        finf_plt = finf.copy()
        finf_plt[idomain1 == 0] = np.nan
        
        fig = plt.figure(figsize=figure_size, dpi=300, tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(finf_plt, cmap="jet")
        title = "Precipitation distribution"
        cbar = plt.colorbar(shrink=0.5)
        cbar.ax.set_title("Infiltration\nrate\nfactor", pad=20)
        plt.xlabel("Column Number")
        plt.ylabel("Row Number")
        fs.heading(heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..",
                "figures",
                "{}{}".format(sim_name + "-finfFact", config.figure_ext),
            )
            fig.savefig(fpth)


# Function that wraps all of the steps for each scenario
#
# 1. build_model,
# 2. write_model,
# 3. run_model, and
# 4. plot_results.
#


def scenario(idx, silent=True):
    sim = build_model(example_name)
    write_model(sim, silent=silent)
    success = run_model(sim, silent=silent)

    if success:
        plot_results(sim, idx)


# nosetest - exclude block from this nosetest to the next nosetest
def test_01():
    scenario(0, silent=False)


# nosetest end

if __name__ == "__main__":
    # ### Mehl and Hill (2013) results
    #
    # Two-dimensional transport in a uniform flow field

    scenario(0)
