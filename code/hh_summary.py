# %%
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# values
# IPCC Assessment Report 6, Working Group I, Chapter 7 The Earth’s Energy Budget, Climate Feedbacks, and Climate Sensitivity
# Section 7.6.1.1 Radiative Properties and Lifetimes, (Table 7.15, url)
EF_CH4 = 29.8
EF_N2O = 273

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
# pivot and get total emissions
e['ghg_name'] = e.ghg_name.str.lower().str.replace(' ', '_', regex=False)
ew = e.pivot(index=['facility_id', 'facility_name', 'reporting_year'],
             columns='ghg_name', values='ghg_quantity').reset_index()
ew['total_emissions'] = (ew.biogenic_carbon_dioxide.fillna(0) + 
                         ew.methane.fillna(0)*EF_CH4 + 
                         ew.nitrous_oxide.fillna(0)*EF_N2O)

# %%
# SUMMARIZE EMISSIONS
# table
summ = (ew.groupby(['reporting_year'])['total_emissions']
        .agg(
            total_emissions='sum',
            total_facilities='count',
            mean='mean',
            std='std',
            min='min',
            pctl25=lambda x: np.percentile(x, 25),
            pctl50=lambda x: np.percentile(x, 50),
            pctl75=lambda x: np.percentile(x, 75),
            max='max'
        ).style.format('{:,.2f}'))
display(summ)
# plot
fig, axl = plt.subplots()
axr = axl.twinx()
grouped = ew.groupby('reporting_year')['total_emissions']
pct_te = grouped.sum().values
pct_te /= pct_te[0]
pct_c = grouped.count().values.astype(float)
pct_c /= pct_c[0]
pct_epf = grouped.mean().values
pct_epf /= pct_epf[0]

# Creating boxplot
axl.boxplot([group for _, group in grouped], labels=grouped.groups.keys(), showfliers=False)
axl.axhline(y=25000, color='black', linewidth=1, linestyle=':', label='emissions reporting cutoff')
axr.plot(range(1, len(grouped.groups) + 1), pct_te, color='C3', label='total emissions')
axr.plot(range(1, len(grouped.groups) + 1), pct_c, color='C2', label='total facilities')
axr.plot(range(1, len(grouped.groups) + 1), pct_epf, color='C4', label='mean emissions per facility')
axr.set_ylabel(f'percent of {min(grouped.groups.keys())} numbers [lineplot]')
axl.set_ylabel('facility emissions [boxplot]')
axl.set_xlabel('reporting year')
plt.title('subpart HH reported emissions')
plt.legend()
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
# manually fix one reporting error I found
l.loc[(l.reporting_year == 2014) & (l.facility_id == 1006107), 'landfill_capacity'] = (
    l.loc[(l.reporting_year == 2013) & (l.facility_id == 1006107), 'landfill_capacity'])

# %%
# READIN DATA
# waste quantity data
filepath = PATH_DATA_RAW + 'hh_ann_waste_disposal_qty.csv'
try:
    q = pd.read_csv(filepath, encoding='utf-8')
except UnicodeDecodeError:
    q = pd.read_csv(filepath, encoding='latin1')
q.columns = q.columns.str.lower()
# clean up method column
q['method'] = np.where(q.method_used_to_find_qty.str.startswith('USED SCALES'), 'm_scales',
                       np.where(q.method_used_to_find_qty == 'OTHER', 'm_other', 'm_working_capacity'))
# pivot and sum
qw = q.pivot(index=['facility_id', 'reporting_year', 'waste_disp_reporting_year', 'total_waste_disposal_qty_ry'],
             columns='method',
             values='waste_disposal_qty_ry_by_mthd').reset_index()
qw['total_waste_disposal'] = (qw.m_other.fillna(0) + qw.m_scales.fillna(0) + qw.m_working_capacity.fillna(0))
# collapse to actual year
qwc = (
    qw.groupby(['facility_id', 'waste_disp_reporting_year'])
    [['total_waste_disposal_qty_ry']]
    .mean().reset_index()
    .rename(columns={'waste_disp_reporting_year':'reporting_year'}))
qwc

# %%
# MERGE DATA
# first merge emissions and properties
# check that datasets align
assert len(l) == len(ew)
el = pd.merge(left=ew, right=l.drop(columns='facility_name'), how='outer',
              on=['facility_id', 'reporting_year'], indicator='_merge_el')
print(el.groupby('_merge_el')['facility_id'].count())
el['emissions_per_capacity'] = el.total_emissions / el.landfill_capacity
el['yrs_until_close'] = el.estimated_yr_of_lndfil_closure - el.reporting_year
el['yrs_from_open'] = el.reporting_year - el.first_yr_lndfil_accepted_waste
el['is_landfill_open'] = el.is_landfill_open == 'Y'
el['has_gas_clct'] = el.does_lndfil_have_gas_clct == 'Y'
el['is_leachate_recirculation_used'] = el.leachate_recirculation_used == 'Y'
el['has_passive_vents_or_flares'] = el.passive_vents_or_flares == 'Y'
el['has_scales'] = el.scales_present == 'Y'

# next merge with quantities
elq = pd.merge(left=el, right=qwc, how='left',
               on=['facility_id', 'reporting_year'], indicator='_merge_elq')
print(elq.groupby('_merge_elq')['facility_id'].count())
elq['emissions_per_waste_qty'] = np.where(elq.total_waste_disposal_qty_ry == 0, np.nan, elq.total_emissions / elq.total_waste_disposal_qty_ry)


# %%
# SUMMARIZE OVERALL
# table
def pct(x):
    return x.sum() / x.count()
summ = (elq.groupby(['reporting_year'], dropna=False)
        .agg({
            'yrs_until_close':['mean', 'std'],
            'yrs_from_open':['mean', 'std'],
            'landfill_capacity':['mean', 'std'],
            'lndfil_surface_containing_wste':['mean', 'std'],
            'is_landfill_open':['sum', pct],
            'has_gas_clct':['sum', pct],
            'is_leachate_recirculation_used':['sum', pct],
            'has_passive_vents_or_flares':['sum', pct],
            'has_scales':['sum', pct],
            'emissions_per_capacity':['mean', 'std'],
            'total_waste_disposal_qty_ry':['mean', 'std']
        })
)
summ.to_csv(PATH_RESULTS + 'df_annual_characteristics.csv')
# plot share of landfill characteristics
fig, axl = plt.subplots()
axr = axl.twinx()
axl.plot(summ[('is_landfill_open', 'pct')].values, label='that are open')
axl.plot(summ[('has_scales', 'pct')].values, label='with scales')
axl.plot(summ[('has_passive_vents_or_flares', 'pct')].values, label='with passive vents or flares')
axl.plot(summ[('has_gas_clct', 'pct')].values, label='with gas collector')
axl.plot(summ[('is_leachate_recirculation_used', 'pct')].values, label='using leachate recirc.')
axl.set_xticks(range(len(summ)))
axl.set_xticklabels(summ.reset_index().reporting_year.values)
axr.plot(summ[('landfill_capacity', 'mean')].values / summ[('landfill_capacity', 'mean')].values[0], 
         color='grey', linewidth=1.0, linestyle='--', alpha=0.75,
         label='mean capacity')
axr.plot(summ[('lndfil_surface_containing_wste', 'mean')].values / summ[('lndfil_surface_containing_wste', 'mean')].values[0], 
         color='black', linewidth=1.0, linestyle='--', alpha=0.75,
         label='mean surface area')
axr.plot(summ[('emissions_per_capacity', 'mean')].values / summ[('emissions_per_capacity', 'mean')].values[0], 
         color='grey', linewidth=1.0, linestyle='-.', alpha=0.75,
         label='mean emissions per capacity')
axr.plot(summ[('total_waste_disposal_qty_ry', 'mean')].values / summ[('total_waste_disposal_qty_ry', 'mean')].values[0], 
         color='black', linewidth=1.0, linestyle='-.', alpha=0.75,
         label='mean waste qty')
axl.set_ylabel('percent of landfills [solid lines]')
axr.set_ylabel(f'percent of {min(grouped.groups.keys())} values [dotted lines]')
axl.set_title('landfill characteristics over time')
axl.legend(loc='center left', bbox_to_anchor=(0,0.085), ncol=2, frameon=False)
axr.legend(loc='upper right', ncol=2, frameon=False)
axl.set_ylim((0, 1))
axr.set_ylim((0, 2))
plt.show()


# %%
# SUMMARIZE NEW AND OLD
el['year_first_reporting'] = el.groupby('facility_id')['reporting_year'].transform('min')
el['is_min_reporting_year'] = el.reporting_year == el.year_first_reporting
el['year_last_reporting'] = el.groupby('facility_id')['reporting_year'].transform('max')
el['is_max_reporting_year'] = el.reporting_year == el.year_last_reporting
el['reporting_year_cat'] = np.where(el.is_max_reporting_year, 
                                    np.where(el.is_min_reporting_year, 
                                             '0 - single year', 
                                             '3 - last reporting year'),
                                    np.where(el.is_min_reporting_year, 
                                             '1 - first reporting year',
                                             '2 - all else'))
summ = (el.groupby('reporting_year', dropna=True)
        .agg(
            n=('facility_id','count'),
            n_first_yr_reporting=('is_min_reporting_year', 'sum'),
            n_last_yr_reporting=('is_max_reporting_year', 'sum')            
        ))
summ['n_skipped_reporting'] = (
    (summ.n.shift(1).fillna(0) +
    summ.n_first_yr_reporting - 
    summ.n_last_yr_reporting.shift(1).fillna(0)) - 
    summ.n
).astype(int)
summ.to_csv(PATH_RESULTS + 'df_facility_counts_by_reporting_year.csv')

summ = (
    el.groupby(['reporting_year', 'reporting_year_cat'])
    .agg({
        'facility_id':['count'],
        'yrs_from_open':['mean'],
        'landfill_capacity':['mean'],
        'lndfil_surface_containing_wste':['mean'],
        'is_landfill_open':[pct],
        'has_gas_clct':[pct],
        'is_leachate_recirculation_used':[pct],
        'has_passive_vents_or_flares':[pct],
        'has_scales':[pct],
        'emissions_per_capacity':['mean']
        })
)
summ.to_csv(PATH_RESULTS + 'df_characteristics_by_reporting_cat.csv')
# %%
