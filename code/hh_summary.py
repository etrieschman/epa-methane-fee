# %%
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# paths
PATH_DATA = '../data/'
PATH_DATA_RAW = PATH_DATA + 'raw/'
PATH_DATA_PROCESSED = PATH_DATA + 'processed/'
PATH_DATA_RESOURCES = PATH_DATA + 'resources/'
PATH_RESULTS = '../results/hh_summary/'
rpath = Path(PATH_RESULTS)
rpath.mkdir(parents=True, exist_ok=True)

# %%
# READIN DATA
# Emissions data
filepath = PATH_DATA_RAW + 'hh_subpart_level_information.csv'
try:
    e = pd.read_csv(filepath, encoding='utf-8')
except UnicodeDecodeError:
    e = pd.read_csv(filepath, encoding='latin1')
e.columns = e.columns.str.lower()
# describe emissions
print('summary by emission type')
print(e.groupby('ghg_name')['ghg_quantity'].describe())

# drop non-methane emissions (they're trivial)
e = e.loc[e.ghg_name == 'METHANE'].drop(columns='ghg_name').rename(columns={'ghg_quantity':'ghg_methane'})
summ = e.groupby(['reporting_year'])['ghg_methane'].agg(['count', 'sum', 'mean'])
# summ['mean'].plot()
print('pct change in mean', summ['mean'].iloc[-1] / summ['mean'].iloc[0] - 1)
print('pct change in facilities', summ['count'].iloc[-1] / summ['count'].iloc[0] - 1)
print('pct change in emissions', summ['sum'].iloc[-1] / summ['sum'].iloc[0] - 1)

plt.boxplot(e.ghg_methane * 29 / 1e6, showfliers=False)
plt.axhline(y=25000/1e6)
plt.show()

# %%
# READIN DATA
# Landfill info
filepath = PATH_DATA_RAW + 'hh_landfill_info.csv'
try:
    l = pd.read_csv(filepath, encoding='utf-8')
except UnicodeDecodeError:
    l = pd.read_csv(filepath, encoding='latin1')
l.columns = l.columns.str.lower()
l.groupby(['reporting_year', 'is_landfill_open']).agg(
    {'facility_id':['count', 'nunique'],
    'annual_modeled_ch4_generation':'sum'}
)