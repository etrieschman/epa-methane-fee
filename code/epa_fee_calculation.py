# %%
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from epa_fee_units import epa_fee_units

# options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# paths
PATH_DATA = '../data/'
PATH_DATA_RAW = PATH_DATA + 'raw/'
PATH_DATA_PROCESSED = PATH_DATA + 'processed/'
PATH_DATA_RESOURCES = PATH_DATA + 'resources/'
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
m_sub = m.drop(columns=[col for col in m.columns if col.startswith('facility')])

# facility info
f = pd.DataFrame()
for year in tqdm(m.reporting_year.unique()):
    xls = pd.ExcelFile(PATH_DATA_RAW + 
                       f'2022_data_summary_spreadsheets/ghgp_data_{year}.xlsx')
    for sheet in (xls.sheet_names):
        if sheet in ['Industry Type', 'FAQs about this Data', 
                     'Suppliers', 'SF6 from Elec. Equip.',
                     'CO2 Injection', 'Geologic Sequestration of CO2',
                     'LDC - Direct Emissions']:
            continue
        fyr = xls.parse(sheet, header=3)
        fyr['reporting_year'] = year
        fyr['segment'] = sheet
        f = pd.concat([fyr, f], ignore_index=True)
f.columns = (f.columns
             .str.lower()
             .str.replace('.', '', regex=False)
             .str.replace('(\(|\)|\/|\,|\-|\–|\&)', '', regex=True)
             .str.strip()
             .str.replace(r'\s+', '_', regex=True)
)
f.rename(columns={'facility_id':'ghgrp_facility_id'}, inplace=True)
# drop direct emitters values if onshore production
# clean addresses
f['facility_cln_city'] = (f.reported_city.str.lower()
                           .fillna(f.city.str.lower()))
f['facility_cln_state'] = (f.reported_state.str.lower()
                            .fillna(f.state.str.lower()))
f['facility_cln_zip'] = (f.reported_zip_code
                          .fillna(f.zip_code))
f['facility_cln_county'] = (f.reported_county.str.lower()
                            .fillna(f.county.str.lower()))
f['facility_cln_address'] = (f.reported_address.str.lower()
                            .fillna(f.address.str.lower()))
f['facility_cln_latitude'] = (f.latitude.fillna(f.reported_latitude))
f['facility_cln_longitude'] = (f.longitude.fillna(f.reported_longitude))

cols_keep = [
    'reporting_year', 
    'ghgrp_facility_id',
    'primary_naics_code'
]
cols_keep += [col for col in f.columns if col.startswith('facility_cln_')]
f_sub = f[cols_keep].drop_duplicates().reset_index()

# merge on
mf = pd.merge(left=m_sub, right=f_sub, how='left',
              on=['ghgrp_facility_id', 'reporting_year'])
print(len(m), len(mf))


# %%
# READIN DATA
# Emissions data
filepath = PATH_DATA_RAW + 'ghgp_data_emissions_2015_2022.csv'
try:
    e = pd.read_csv(filepath, encoding='utf-8')
except UnicodeDecodeError:
    e = pd.read_csv(filepath, encoding='latin1')
# e = pd.read_csv(PATH_DATA_RAW + 'ghgp_data_emissions_2015_2022.csv')
e.columns = e.columns.str.lower()
e.rename(columns={'facility_id':'ghgrp_facility_id'}, inplace=True)
e_col = (
    e.groupby(['ghgrp_facility_id', 'facility_name', 'reporting_year', 
               'industry_segment', 'basin_associated_with_facility'],
              dropna=False)
    [['total_reported_co2_emissions', 'total_reported_ch4_emissions',
      'total_reported_n2o_emissions']]
    .sum()
    .reset_index()
)
e_col['total_reported_emissions'] = (
    e_col.total_reported_co2_emissions.fillna(0) + 
    e_col.total_reported_ch4_emissions.fillna(0) + 
    e_col.total_reported_n2o_emissions.fillna(0)
)


# %%
# READIN DATA
# Production data
p = pd.read_csv(PATH_DATA_RAW + 'ghgp_data_production_2015_2022.csv')
p.rename(columns={'facility_id':'ghgrp_facility_id'}, inplace=True)
p['counts'] = p.groupby(['ghgrp_facility_id', 'reporting_year'], dropna=False)['ghgrp_facility_id'].transform('count')
p_sub = p.loc[(~p.table_num.isin(['Table AA.1.ii', 'Table AA.10.i']))]

# merge on methane fee units
pu = pd.merge(left=p_sub, right=epa_fee_units, how='left',
              on='industry_segment')
print('Lenghths pre/post merge:', len(p_sub), len(pu))

# unify quantity and thresholds
conditions = [
    pu.segment_abr == 'w-p-offsh',
    pu.segment_abr == 'w-ng-offsh',
    pu.segment_abr == 'w-p-onsh',
    pu.segment_abr == 'w-ng-onsh',
    pu.segment_abr == 'w-ng-ust',
    pu.segment_abr == 'w-lng-st',
    pu.segment_abr == 'w-lng-ei',
    pu.segment_abr == 'w-ng-gb',
    pu.segment_abr == 'w-ng-trans',
    pu.segment_abr == 'w-ng-tc',
    pu.segment_abr == 'w-ng-proc'
]
choices_quant = [
    pu.quantity_of_oil_handled.fillna(0),
    pu.quantity_of_gas_handled.fillna(0),
    pu.oil_prod_cal_year_for_sales.fillna(0),
    pu.gas_prod_cal_year_for_sales.fillna(0),
    (pu.quantity_of_gas_withdrawn.fillna(0) + pu.quantity_gas_withdrawn.fillna(0)),
    (pu.quantity_lng_withdrawn.fillna(0)),
    (pu.quantity_exported.fillna(0) + pu.quantity_imported.fillna(0)),
    (pu.quant_gas_transported_gb.fillna(0) + pu.quant_hc_liq_trans.fillna(0)),
    (pu.quantity_gas_transferred.fillna(0)),
    (pu.quantity_gas_transferred.fillna(0)),
    (pu.quantity_of_gas_handled.fillna(0))
]

pu['throughput'] = np.select(conditions, choices_quant, default=pd.NA)
pu['emission_threshold'] = pu.throughput * pu.waste_em_thresh * pu.unit_conversion
pu_col = (
    pu.groupby([
        'ghgrp_facility_id', 'reporting_year', 
        'industry_segment', 'industry_segment_abr',
        'basin_associated_with_facility'], dropna=False)
    [['throughput', 'emission_threshold']]
    .sum()
    .reset_index()
)
# %%
# MERGE PRODUCTION AND EMISSIONS
pe = pd.merge(left=e_col, right=pu_col, how='outer',
              on=['ghgrp_facility_id', 'reporting_year', 'industry_segment'],
              indicator=True)

# check for duplicates
pe['counts'] = pe.groupby(['ghgrp_facility_id', 'reporting_year', 'industry_segment', '_merge'], dropna=False)['ghgrp_facility_id'].transform('count')
assert len(pe.loc[pe.counts > 1]) == 0
# clean up basin info
pe['basin'] = pe.basin_associated_with_facility_x.fillna(
                pe.basin_associated_with_facility_y)
pe.loc[(pe.basin == ' ') | pe.basin.isna(), 'basin'] = np.nan
pe['basin_num'] = pd.to_numeric(pe.basin.str.replace('[^0-9]', '', regex=True))
# drop distribution category
pe = pe.loc[pe.industry_segment != 'Natural gas distribution [98.230(a)(8)]']
pe['applicable_emissions'] = pe.total_reported_ch4_emissions - pe.emission_threshold
pe['applicable_emissions'] = np.where(pe.throughput == 0, pd.NA, pe.applicable_emissions)
pe['report_gt25k'] = pe.total_reported_emissions > 25000
cols = ['applicable_emissions', 'throughput', 'emission_threshold', 
        'total_reported_emissions', 'total_reported_ch4_emissions']
for col in cols:
    pe[f'{col}_gt25k'] = np.where(pe.report_gt25k, pe[col], pd.NA)

abr = epa_fee_units[['industry_segment', 'industry_segment_abr']].drop_duplicates()
pe = pd.merge(left=pe.drop(columns='industry_segment_abr'), right=abr,
              how='left', on='industry_segment')


# %%
def median_percentile(x):
    if len(x.dropna()) > 0:
        return np.percentile(x.dropna(), 50)
    else:
        return np.nan
def iqr(x, round=2):
    if len(x.dropna()) > 0:
        # Calculate percentiles and round
        lower_percentile = np.percentile(x.dropna(), 25).round(round)
        upper_percentile = np.percentile(x.dropna(), 75).round(round)
        # Format with commas
        formatted_lower = '{:,}'.format(lower_percentile)
        formatted_upper = '{:,}'.format(upper_percentile)

        out = f'[{formatted_lower}-{formatted_upper}]'

        return out
    else:
        return np.nan

summ = (
    pe
    .loc[pe.reporting_year >= 2021]
    .groupby(['industry_segment_abr', 'reporting_year'])
    .agg(
        n=('ghgrp_facility_id', 'count'),
        n_gt25k=('report_gt25k','sum'),
        med_te=('total_reported_emissions', lambda x: median_percentile(x)),
        med_te_gt25=('total_reported_emissions_gt25k', lambda x: median_percentile(x)),
        iqr_te_gt25=('total_reported_emissions_gt25k', lambda x: iqr(x)),
        med_ch4e=('total_reported_ch4_emissions', lambda x: median_percentile(x)),
        med_ch4e_gt25=('total_reported_ch4_emissions_gt25k', lambda x: median_percentile(x)),
        iqr_ch4e_gt25=('total_reported_ch4_emissions_gt25k', lambda x: iqr(x)),
        med_th=('emission_threshold', lambda x: median_percentile(x)),
        med_th_gt25=('emission_threshold_gt25k', lambda x: median_percentile(x)),
        iqr_th_gt25=('emission_threshold_gt25k', lambda x: iqr(x)),
        med_ae=('applicable_emissions', lambda x: median_percentile(x)),
        med_ae_gt25=('applicable_emissions_gt25k', lambda x: median_percentile(x)),
        iqr_ae_gt25=('applicable_emissions_gt25k', lambda x: iqr(x)),
    )
    .reset_index()
)
summ.to_csv(PATH_RESULTS + 'df_summary_pctl.csv', index=False)

# %%
# Create a 3x3 grid of subplots
var = 'total_reported_emissions'
fig, axes = plt.subplots(3, 3, sharex=False, figsize=(10, 8))
axes = axes.ravel()  # Flatten the array for easy indexing
# Loop through segments and create a histogram for each
for i, segment in enumerate(pe.industry_segment_abr.unique()):
    # Select data for the segment
    segment_data = pe[pe.industry_segment_abr == segment][var]
    # Plot the histogram
    axes[i].hist(segment_data.dropna(), bins=30, color=f'C{i}', density=True, alpha=0.8)
    axes[i].axvline(x=25000, color='red', linestyle=':', linewidth=1)
    axes[i].set_title(segment)
    axes[i].set_yscale('log')
axes[7].set_xlabel(var)

# Adjust layout for clarity
plt.tight_layout()
plt.savefig(PATH_RESULTS + f'fig_hist_{var}', bbox_inches='tight', dpi=300)

# %%
# Get devon facilities
mf_dev = mf.loc[m.parent_company_name.notna()]
mf_dev = mf_dev.loc[mf_dev.parent_company_name.str.lower().str.contains('devon')]
print(mf_dev.groupby(['reporting_year'])[['ghgrp_facility_id']].agg(['count', 'nunique']))

# subset emissions data
pe_dev = pd.merge(left=mf_dev, right=pe, how='left',
                 on=['ghgrp_facility_id', 'reporting_year'], 
                 indicator='merge_parent')

print(pe_dev.groupby('merge_parent')['ghgrp_facility_id'].nunique())

# %% 
# SUMMARIZE DEVON OPERATIONS
# MAP OF OPERATIONS
import geopandas as gpd
import matplotlib.pyplot as plt
year = 2022

# Read in the US states shapefile using GeoPandas
gdf_states = gpd.read_file(PATH_DATA_RESOURCES + 'cb_state_boundaries/cb_2018_us_state_20m.shp')
gdf_states = gdf_states.loc[~gdf_states.STUSPS.isin(['HI', 'AK', 'PR'])]
gdf_basins = gpd.read_file(PATH_DATA_RESOURCES + 'Basins_Shapefile/Basins_GHGRP.shp')

# Convert the DataFrame to a GeoDataFrame
gpe_dev = pe_dev.loc[pe_dev.reporting_year == year]
gdf = gpd.GeoDataFrame(
    gpe_dev, geometry=gpd.points_from_xy(
        gpe_dev.facility_cln_longitude, 
        gpe_dev.facility_cln_latitude)
        )
fig, ax = plt.subplots(1, 1, figsize=(6, 8))
gdf_states.geometry.plot(ax=ax, linewidth=0.8, color=None, alpha=0.5, edgecolor='black')
gdf_basins.plot(ax=ax, linewidth=0.05, cmap='Blues', column='BASIN_NAME', alpha=0.5)
gdf.plot(ax=ax, 
         column='industry_segment', cmap='viridis_r',
         legend=True, 
         markersize=gdf.total_reported_emissions.values / 1e3,
         alpha=0.75)
leg = ax.get_legend()
leg.set_bbox_to_anchor((0.5, -0.15))
leg.set_loc('upper center')
plt.xlim((-110, -90))
plt.ylim((27, 42))
plt.title(f'Company A operations in {year}')
plt.show()

# SUMMARY OF OPERATIONS
summ = pe_dev.loc[pe_dev.reporting_year == year, [
    'industry_segment_abr', 'basin_num',
    'ghgrp_facility_id', 'facility_name',
    'parent_co_percent_ownership',
    'facility_cln_state', 'facility_cln_city'
]]
display(summ)
summ.to_csv(PATH_RESULTS + 'df_devon_facilities.csv', index=False)

summ = pe_dev.loc[pe_dev.reporting_year == year, [
    'industry_segment_abr', 
    'basin_num',
    'ghgrp_facility_id',
    'total_reported_emissions',
    'total_reported_ch4_emissions', 
    'throughput', 'emission_threshold', 'applicable_emissions'
]]
display(summ)
summ.to_csv(PATH_RESULTS + 'df_devon_fee.csv', index=False)

# save lookup table values
lead_cols = ['industry_segment']
lag_cols = [col for col in epa_fee_units.columns if col not in lead_cols]
epa_fee_units[lead_cols + lag_cols].to_csv(PATH_RESULTS + 'key_units.csv', index=False)


# %%
