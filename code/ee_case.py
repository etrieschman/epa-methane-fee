# %%
import pandas as pd

# paths
PATH_DATA = '../data/'
PATH_DATA_RAW = PATH_DATA + 'raw/'

df = pd.read_parquet(PATH_DATA_RAW + 'model_ready.parquet')
# %%
df
# %%
df.columns
# %%
