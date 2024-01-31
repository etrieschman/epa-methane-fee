import pandas as pd
epa_fee_dict = {
    'w-ng-proc':{
        'industry_segment_abr':'w-ng-proc',
        'industry_segment':'Onshore natural gas processing [98.230(a)(3)]',
        'waste_em_thresh':0.0005, # Mscf CH4 / Mscf natgas
        'unit_conversion':0.0192 # mt / Mscf natgas
    },
    'w-ng-tc':{
        'industry_segment_abr':'w-ng-tc',
        'industry_segment':'Onshore natural gas transmission compression [98.230(a)(4)]',
        'waste_em_thresh':0.0011, #Msch CH4 / Mscf natgas
        'unit_conversion':0.0192 # mt / Mscf natgas
    },
    'w-ng-ust':{
        'industry_segment_abr':'w-ng-ust',
        'industry_segment':'Underground natural gas storage [98.230(a)(5)]',
        'waste_em_thresh':0.0011, #Msch CH4 / Mscf natgas
        'unit_conversion':0.0192 # mt / Mscf natgas
    },
    'w-ng-onsh':{
        'industry_segment_abr':'w-onsh',
        'industry_segment':'Onshore petroleum and natural gas production [98.230(a)(2)]',
        'waste_em_thresh':0.002, # Mscf CH4 / Mscf natgas
        'unit_conversion':0.0192 # mt / Mscf natgas
    },
    'w-p-onsh':{
        'industry_segment_abr':'w-onsh',
        'industry_segment':'Onshore petroleum and natural gas production [98.230(a)(2)]',
        'waste_em_thresh':10, # mt CH4 / mbarrels oil
        'unit_conversion': 1e-6 # barrels / mbarrels
    },
    'w-ng-offsh':{
        'industry_segment_abr':'w-offsh',
        'industry_segment':'Offshore petroleum and natural gas production [98.230(a)(1)]',
        'waste_em_thresh':0.002, # Mscf CH4 / Mscf natgas
        'unit_conversion':0.0192 # mt / Mscf natgas
    },
    'w-p-offsh':{
        'industry_segment_abr':'w-offsh',
        'industry_segment':'Offshore petroleum and natural gas production [98.230(a)(1)]',
        'waste_em_thresh':10, # mt CH4 / mbarrels oil
        'unit_conversion': 1e-6 # barrels / mbarrels
    },
    'w-lng-st':{
        'industry_segment_abr':'w-lng-st',
        'industry_segment':'Liquefied natural gas (LNG) storage [98.230(a)(6)]',
        'waste_em_thresh':0.0005, # Mscf CH4 / Mscf natgas
        'unit_conversion':0.0192 # mt / Mscf natgas
    },
    'w-lng-ei':{
        'industry_segment_abr':'w-lng-ei',
        'industry_segment':'LNG import and export equipment [98.230(a)(7)]',
        'waste_em_thresh':0.0005, # Mscf CH4 / Mscf natgas
        'unit_conversion':0.0192 # mt / Mscf natgas
    },
    'w-gb':{
        'industry_segment_abr':'w-gb',
        'industry_segment':'Onshore petroleum and natural gas gathering and boosting [98.230(a)(9)]',
        'waste_em_thresh':0.0005, # Mscf CH4 / Mscf natgas
        'unit_conversion':0.0192 # mt / Mscf natgas
    },
    'w-ng-trans':{
        'industry_segment_abr':'w-ng-trans',
        'industry_segment':'Onshore natural gas transmission pipeline [98.230(a)(10)]',
        'waste_em_thresh':0.0011, #Msch CH4 / Mscf natgas
        'unit_conversion':0.0192 # mt / Mscf natgas
    },
}

epa_fee_units = pd.DataFrame(epa_fee_dict).T.reset_index(names='segment_abr')