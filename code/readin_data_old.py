# %%


# %%
# READIN DATA
# Emissions data
# FROM A DIFFERENT SOURCE, HARDER TO WORK WITH
# e = pd.DataFrame()
# for year in tqdm(m.reporting_year.unique()):
#     xls = pd.ExcelFile(PATH_DATA_RAW + 
#                        f'2022_data_summary_spreadsheets/ghgp_data_{year}.xlsx')
#     for sheet in (xls.sheet_names):
#         if sheet in ['Industry Type', 'FAQs about this Data', 
#                      'Suppliers', 'SF6 from Elec. Equip.',
#                      'CO2 Injection', 'Geologic Sequestration of CO2',
#                      'LDC - Direct Emissions']:
#             continue
#         eyr = xls.parse(sheet, header=3)
#         eyr['reporting_year'] = year
#         eyr['segment'] = sheet
#         e = pd.concat([eyr, e], ignore_index=True)
# e.columns = (e.columns
#              .str.lower()
#              .str.replace('.', '', regex=False)
#              .str.replace('(\(|\)|\/|\,|\-|\–|\&)', '', regex=True)
#              .str.strip()
#              .str.replace(r'\s+', '_', regex=True)
# )
# e.rename(columns={'facility_id':'ghgrp_facility_id'}, inplace=True)

# # %%
# # CLEAN UP DATA
# # Split the 'industry_type_subparts' column and get unique subparts
# e['industry_type_subparts_list'] = (
#     e['industry_type_subparts']
#     .str.replace(r'\(.*?\)', '', regex=True)
#     .str.replace(' ', '', regex=False)
#     .str.lower()
#     .str.split(',')
# )
# e['industry_type_subpart_w'] = (
#     e.industry_type_subparts_list.apply(lambda x: ','.join(i for i in x if i.startswith('w')))
# )
# all_subparts = set().union(*e.industry_type_subparts_list.values)
# w_subparts = {i for i in all_subparts if i.startswith('w')}

# # Create boolean columns for each subpart
# for subpart in w_subparts:
#     column_name = f'subpart_{subpart}'
#     e[column_name] = e['industry_type_subparts_list'].apply(lambda x: subpart in x)

# # subset to all facilities related to subpart w
# subpart_ws = [col for col in e.columns if col.startswith('subpart_')]
# e_sub = e.loc[e[subpart_ws].any(axis=1)]
# e_sub = e_sub.dropna(axis=1, how='all')
# # drop direct emitters values if onshore production
# e_sub = e_sub.loc[~(e_sub['subpart_w-onsh'] & (e_sub.segment == 'Direct Emitters'))]
# # examine repeats
# e_sub['counts'] = e_sub.groupby(['ghgrp_facility_id', 'reporting_year', 'state_where_emissions_occur'], dropna=False)['ghgrp_facility_id'].transform('count')
# assert len(e_sub.loc[(e_sub.counts > 1) ].sort_values(['ghgrp_facility_id', 'reporting_year'])) == 0

# # coalese emissions column
# conditions = [
#     (e_sub.industry_type_subpart_w == 'w-onsh'),
#     (e_sub.industry_type_subpart_w == 'w-gb'),
#     (e_sub.industry_type_subpart_w == 'w-trans'),
#     (e_sub.segment == 'Direct Emitters')
# ]
# choices = [
#     e_sub.total_reported_emissions_from_onshore_oil_gas_production,
#     e_sub.total_reported_emissions_from_gathering_boosting,
#     e_sub.total_reported_direct_emissions_from_transmission_pipelines,
#     (e_sub['subpart_w-proc'] * 
#      e_sub.petroleum_and_natural_gas_systems_processing).fillna(0) + 
#     (e_sub['subpart_w-ngtc'] * 
#      e_sub.petroleum_and_natural_gas_systems_transmissioncompression).fillna(0) + 
#     (e_sub['subpart_w-offsh'] * 
#      e_sub.petroleum_and_natural_gas_systems_offshore_production).fillna(0) +
#     (e_sub['subpart_w-unstg'] * 
#      e_sub.petroleum_and_natural_gas_systems_underground_storage).fillna(0) + 
#     (e_sub['subpart_w-lngie'] * 
#      e_sub.petroleum_and_natural_gas_systems_lng_importexport).fillna(0) +
#     (e_sub['subpart_w-lngstg'] * 
#      e_sub.petroleum_and_natural_gas_systems_lng_storage).fillna(0),
# ]

# e_sub['subpart_w_emissions'] = np.select(conditions, choices, default=pd.NA)
# display(e_sub.groupby('industry_type_subpart_w')['subpart_w_emissions'].agg(['sum', 'count']))

# # clean addresses
# e_sub['facility_cln_city'] = (e_sub.reported_city.str.lower()
#                            .fillna(e_sub.city.str.lower()))
# e_sub['facility_cln_state'] = (e_sub.reported_state.str.lower()
#                             .fillna(e_sub.state.str.lower()))
# e_sub['facility_cln_zip'] = (e_sub.reported_zip_code
#                           .fillna(e_sub.zip_code))
# e_sub['facility_cln_county'] = (e_sub.reported_county.str.lower()
#                             .fillna(e_sub.county.str.lower()))
# e_sub['facility_cln_address'] = (e_sub.reported_address.str.lower()
#                             .fillna(e_sub.address.str.lower()))
# e_sub['facility_cln_latitude'] = (e_sub.latitude.fillna(e_sub.reported_latitude))
# e_sub['facility_cln_longitude'] = (e_sub.longitude.fillna(e_sub.reported_longitude))

# cols_keep = [
#     'reporting_year', 
#     'ghgrp_facility_id', 'frs_id', 'facility_name',
#     'basin',
#     'industry_type_subparts_list',
#     'industry_type_subpart_w',
#     'subpart_w_emissions', 'methane_ch4_emissions'
# ]
# cols_keep += [col for col in e_sub.columns if col.startswith('facility_cln_')]
# cols_keep += [col for col in e_sub.columns if col.startswith('subpart_')]
# e_sub = e_sub[cols_keep]
# 'w-offsh'  
# 'w-onsh':,
# 'w-proc':,
# 'w-ngtc':,
# 'w-unstg':,
# 'w-lngstg':,
# 'w-lngie':,
# 'w-gb':,
# 'w-trans'




