# %%
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

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
m = pd.DataFrame()
for sheet in tqdm(xls.sheet_names):
    myr = xls.parse(sheet)
    # pyr['year'] = int(sheet)
    m = pd.concat([myr, m], ignore_index=True)
m.columns = (m.columns
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
for year in tqdm(m.reporting_year.unique()):
    xls = pd.ExcelFile(PATH_DATA_RAW + 
                       f'2022_data_summary_spreadsheets/ghgp_data_{year}.xlsx')
    for sheet in (xls.sheet_names):
        if sheet in ['Industry Type', 'FAQs about this Data', 
                     'Suppliers', 'SF6 from Elec. Equip.',
                     'CO2 Injection', 'Geologic Sequestration of CO2',
                     'LDC - Direct Emissions']:
            continue
        eyr = xls.parse(sheet, header=3)
        eyr['reporting_year'] = year
        eyr['segment'] = sheet
        e = pd.concat([eyr, e], ignore_index=True)
e.columns = (e.columns
             .str.lower()
             .str.replace('.', '', regex=False)
             .str.replace('(\(|\)|\/|\,|\-|\–|\&)', '', regex=True)
             .str.strip()
             .str.replace(r'\s+', '_', regex=True)
)
e.rename(columns={'facility_id':'ghgrp_facility_id'}, inplace=True)

# %%
# CLEAN UP DATA
# Split the 'industry_type_subparts' column and get unique subparts
e['industry_type_subparts_list'] = (
    e['industry_type_subparts']
    .str.replace(r'\(.*?\)', '', regex=True)
    .str.replace(' ', '', regex=False)
    .str.lower()
    .str.split(',')
)
e['industry_type_subpart_w'] = (
    e.industry_type_subparts_list.apply(lambda x: ','.join(i for i in x if i.startswith('w')))
)
all_subparts = set().union(*e.industry_type_subparts_list.values)
w_subparts = {i for i in all_subparts if i.startswith('w')}

# Create boolean columns for each subpart
for subpart in w_subparts:
    column_name = f'subpart_{subpart}'
    e[column_name] = e['industry_type_subparts_list'].apply(lambda x: subpart in x)

# subset to all facilities related to subpart w
subpart_ws = [col for col in e.columns if col.startswith('subpart_')]
e_sub = e.loc[e[subpart_ws].any(axis=1)]
e_sub = e_sub.dropna(axis=1, how='all')
# drop direct emitters values if onshore production
e_sub = e_sub.loc[~(e_sub['subpart_w-onsh'] & (e_sub.segment == 'Direct Emitters'))]
# examine repeats
e_sub['counts'] = e_sub.groupby(['ghgrp_facility_id', 'reporting_year', 'state_where_emissions_occur'], dropna=False)['ghgrp_facility_id'].transform('count')
e_sub.loc[(e_sub.counts > 1) ].sort_values(['ghgrp_facility_id', 'reporting_year'])

# coalese emissions column
e_sub['total_emissions'] = (
    e_sub.total_reported_emissions_from_onshore_oil_gas_production
    .fillna(e_sub.total_reported_emissions_from_gathering_boosting)
    .fillna(e_sub.total_reported_direct_emissions_from_transmission_pipelines)
    .fillna(e_sub.total_reported_direct_emissions))

# 'w-offsh'  
# 'w-onsh':,
# 'w-proc':,
# 'w-ngtc':,
# 'w-unstg':,
# 'w-lngstg':,
# 'w-lngie':,
# 'w-gb':,
# 'w-trans'


# %%
# READIN DATA
# Production data
p = pd.read_csv(PATH_DATA_RAW + 'ghgp_data_production_2015_2022.csv')
p.rename(columns={'facility_id':'ghgrp_facility_id'}, inplace=True)
# p_sub = p.loc[~p.table_num.isin(['Table AA.1.ii'])]
p['counts'] = p.groupby(['ghgrp_facility_id', 'reporting_year'], dropna=False)['ghgrp_facility_id'].transform('count')

# unify quantity
conditions = [
    p.industry_segment.str.startswith('Offshore petroleum and natural gas production'),
    p.industry_segment.str.startswith('Onshore petroleum and natural gas production'),
    p.industry_segment.str.startswith('Underground natural gas storage'),
    p.industry_segment.str.startswith('Liquefied natural gas (LNG) storage'),
    p.industry_segment.str.startswith('LNG import and export equipment'),
    p.industry_segment.str.startswith('Onshore petroleum and natural gas gathering and boosting'),
    p.industry_segment.str.startswith('Onshore natural gas transmission pipeline')
]
choices = [
    (p.quantity_of_oil_handled.fillna(0) + p.quantity_of_gas_handled.fillna(0)),
    (p.gas_prod_cal_year_for_sales.fillna(0) + p.oil_prod_cal_year_for_sales.fillna(0)),
    (p.quantity_of_gas_withdrawn.fillna(0) + p.quantity_gas_withdrawn.fillna(0)),
    (p.quantity_lng_withdrawn.fillna(0)),
    (p.quantity_exported.fillna(0) + p.quantity_imported.fillna(0)),
    (p.quant_gas_transported_gb.fillna(0) + p.quant_hc_liq_trans.fillna(0)),
    (p.quantity_gas_transferred.fillna(0))
]

p['quantity'] = np.select(conditions, choices, default=pd.NA)
display(p.groupby('industry_segment')['quantity'].sum())
p_sub = p.loc[~p.table_num.isin(['Table AA.1.ii'])]


# %%
# Get devon facilities
m_dev = m.loc[m.parent_company_name.notna()]
m_dev = m_dev.loc[m_dev.parent_company_name.str.lower().str.contains('devon')]
display(m_dev.groupby(['reporting_year'])[['ghgrp_facility_id']].agg(['count', 'nunique']))

# subset emissions data
e_dev = pd.merge(left=m_dev, right=e_sub, how='inner',
                 on=['ghgrp_facility_id', 'reporting_year'], 
                 indicator=False)
# p_dev = pd.merge(left=m_dev, right=p_sub, how='inner',
#                  on=['ghgrp_facility_id', 'reporting_year'], 
#                  indicator=False)

pe_dev = pd.merge(left=e_dev, right=p_sub, how='left',
                  on=['ghgrp_facility_id', 'reporting_year'],
                  indicator=True)
display(pe_dev.groupby('_merge')['ghgrp_facility_id'].nunique())

summ = (pe_dev.loc[pe_dev.reporting_year >= 2020]
        .groupby(['reporting_year', 'ghgrp_facility_id', 'facility_name', 'basin', 'industry_segment'], dropna=False)
        [['quantity', 'total_emissions', 'methane_ch4_emissions']]
        .sum()
        .reset_index())

summ.to_csv(PATH_RESULTS + 'df_devon_q_and_e.csv', index=False)













# %%
# MERGE DATA
pe = pd.merge(left=p, right=e_sub, how='outer',
            on=['ghgrp_facility_id', 'reporting_year'],
            indicator='_merge_pe')
# check merge
print(pe.groupby(['_merge_pe'], observed=False)['ghgrp_facility_id'].nunique())
   

# %%

pep_sub = pd.merge(left=pe_sub, right=pp, how='outer',
               on=['ghgrp_facility_id', 'reporting_year'],
               indicator='_merge_pep')
print(pep_sub.groupby(['_merge_pep'], observed=False)['ghgrp_facility_id'].nunique())

# clean addresses
pep_sub['facility_city_cln'] = (pep_sub.facility_city.str.lower()
                           .fillna(pep_sub.reported_city.str.lower())
                           .fillna(pep_sub.city.str.lower()))
pep_sub['facility_state_cln'] = (pep_sub.facility_state.str.lower()
                            .fillna(pep_sub.reported_state.str.lower())
                            .fillna(pep_sub.state.str.lower()))
pep_sub['facility_zip_cln'] = (pep_sub.facility_zip
                          .fillna(pep_sub.reported_zip_code)
                          .fillna(pep_sub.zip_code))
pep_sub['facility_county_cln'] = (pep_sub.facility_county.str.lower()
                            .fillna(pep_sub.reported_county.str.lower())
                            .fillna(pep_sub.county.str.lower()))
pep_sub['facility_address_cln'] = (pep_sub.facility_address.str.lower()
                            .fillna(pep_sub.reported_address.str.lower())
                            .fillna(pep_sub.address.str.lower()))

cols_keep = [col for col in pep_sub.columns if col.startswith('facility')]













# %%
