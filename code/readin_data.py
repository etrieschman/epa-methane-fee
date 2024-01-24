# %%
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)

# paths
PATH_DATA = '../data/'
PATH_DATA_RAW = PATH_DATA + 'raw/'
PATH_DATA_PROCESSED = PATH_DATA + 'processed/'
PATH_RESULTS = '../results/readin_data/'
rpath = Path(PATH_RESULTS)
rpath.mkdir(parents=True, exist_ok=True)


# %%
# READIN DATA
# Parent:facility mapping
xls = pd.ExcelFile(PATH_DATA_RAW + 'ghgp_data_parent_company_09_2023.xlsb')
p = pd.DataFrame()
for sheet in tqdm(xls.sheet_names):
    pyr = xls.parse(sheet)
    # pyr['year'] = int(sheet)
    p = pd.concat([pyr, p], ignore_index=True)
p.columns = (p.columns
             .str.lower()
             .str.replace('.', '', regex=False)
             .str.replace('(\(|\)|\/|\,|\-|\&)', '', regex=True)
             .str.strip()
             .str.replace(' ', '_', regex=False)
)

# %%
# READIN DATA
# Emissions data
e = pd.DataFrame()
for year in tqdm(p.reporting_year.unique()):
    xls = pd.ExcelFile(PATH_DATA_RAW + 
                       f'2022_data_summary_spreadsheets/ghgp_data_{year}.xlsx')
    for sheet in (xls.sheet_names):
        if sheet in ['Industry Type', 'FAQs about this Data']:
            continue
        eyr = xls.parse(sheet, header=3)
        eyr['reporting_year'] = year
        eyr['segment'] = sheet
        e = pd.concat([eyr, e], ignore_index=True)
e.columns = (e.columns
             .str.lower()
             .str.replace('.', '', regex=False)
             .str.replace('(\(|\)|\/|\,|\-|\&)', '', regex=True)
             .str.strip()
             .str.replace(' ', '_', regex=False)
)
             

# %%
# Split the 'industry_type_subparts' column and get unique subparts
e['industry_type_subparts_list'] = (
    e['industry_type_subparts']
    .str.replace(r'\(.*?\)', '', regex=True)
    .str.replace(' ', '', regex=False)
    .str.lower()
    .str.split(',')
)
all_subparts = set().union(*e.industry_type_subparts_list.values)
all_subparts.remove('')

# Create boolean columns for each subpart
for subpart in all_subparts:
    column_name = f'industry_type_subpart_{subpart}'
    e[column_name] = e['industry_type_subparts_list'].apply(lambda x: subpart in x)

    
# %%
# MERGE DATA
pe = pd.merge(left=p, right=e, how='left',
            left_on=['ghgrp_facility_id', 'reporting_year'],
            right_on=['facility_id', 'reporting_year'],
            indicator=True)

# clean addresses
pe['facility_city_cln'] = pe.facility_city.str.lower().fillna(pe.city.str.lower())
pe['facility_state_cln'] = pe.facility_state.str.lower().fillna(pe.state.str.lower())
pe['facility_zip_cln'] = pe.facility_zip.fillna(pe.zip_code)
pe['facility_county_cln'] = pe.facility_county.str.lower().fillna(pe.county.str.lower())
pe['facility_address_cln'] = pe.facility_address.str.lower().fillna(pe.address.str.lower())

# check merge
print(pe.groupby(['_merge'], observed=False)['ghgrp_facility_id'].count())

# subset to all facilities related to subpart w
subpart_ws = ['w', 'w-gb','w-ldc','w-lngie','w-lngstg',
              'w-ngtc','w-offsh','w-onsh','w-proc','w-trans',
              'w-unstg']
subpart_ws = [f'subpart_{w}' for w in subpart_ws]
pe_sub = pe.loc[pe[subpart_ws].any(axis=1)]
pe_sub = pe_sub.dropna(axis=1, how='all')


# %%
# FIND WHAT THEY'RE UNIQUE ON
unique_on = ['parent_company_name', 
              'reporting_year',
              'basin',
              'industry_type_sectors',
              'facility_naics_code',
              'facility_zip_cln',
              'facility_state_cln', 
              'latitude', 'longitude'
              ]
unique_on += [col for col in pe_sub.columns if 
              col.startswith('subpart_')]
check_unique = (
    pe_sub.groupby(unique_on, dropna=False)
    .agg({'ghgrp_facility_id':['nunique']})
)
pe_sub.loc[:, 'nunique_on_cat'] = pe_sub.groupby(unique_on, dropna=False)['ghgrp_facility_id'].transform('nunique')

# %%
check_unique.value_counts(dropna=False, sort=False)

# %%
pe_sub.loc[pe_sub.nunique_on_cat == 8].sort_values(unique_on)












# %%
# Get devon facilities
# from parent company mapping
p_sub = p.loc[p.parent_company_name.notna()]
p_dev = p_sub.loc[p_sub.parent_company_name.str.lower().str.contains('devon')]
p_dev.groupby(['reporting_year'])[['ghgrp_facility_id', 'frs_id_facility']].agg(['count', 'nunique'])

# from emissions data
e_dev = pd.merge(left=p_dev, right=e, how='left',
                 left_on=['ghgrp_facility_id', 'reporting_year'],
                 right_on=['facility_id', 'reporting_year'])
e_dev = e_dev.dropna(axis=1, how='all')

# %%
