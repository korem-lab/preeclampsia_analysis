import os 
CODE_DIR = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
    
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA
# from deicode.rpca import rpca
from scipy.stats import pearsonr
import biom



def rescale(micro):
     return( micro/(micro.sum(axis=1)[:, np.newaxis] ) )

from skbio.stats.composition import clr
def mv_pseudocount(tbl):
    return np.power(10, np.floor(np.log10(rescale(tbl)[tbl>0].min().min())))



def load_data(data_path=os.path.join(CODE_DIR,'../../data/')):
    
    micro=pd.read_csv(
                os.path.join(data_path, 
                             'numom_hgf_kraken2_VMGC_species_abun_sub500K.csv'), 
                      index_col=0
                     )

    rcs=pd.read_csv(os.path.join(data_path, 
                                 'numom_hgf_PEC_subsampled_read_counts.csv'), 
                index_col=0).loc[micro.index]

    micro=micro.loc[rcs['500K_reads']>=500000.0]
    micro['unmapped'] = 500000.0 - micro.sum(axis=1)
    
    md=pd.read_csv(os.path.join(data_path, 
                                'md_124.csv'), 
                   index_col=0).reset_index()
    
    md.index=md['DNAVisit_1']

    md=md.loc[md.index.isin(micro.index)]
    

    luminex=pd.read_csv(os.path.join(data_path, 
                                     'luminex_lod_by_batch_124.csv'),
                        index_col=0).rename(index={v:k for k,v in md.level_0.to_dict().items()})

    luminex=luminex.loc[md.index]
    
    

    micro=micro.loc[md.index]
    
    ## prevalence based filtering
    micro=micro.loc[:, ( rescale(micro) > 0).mean(axis=0) > 0.1]
    print(micro.shape)
    micro_relabund = micro/micro.values.sum(axis=1)[:, np.newaxis]
    

	### BEFORE:
    #clin=pd.read_csv(os.path.join(data_path,
                                #  'clin_reduced_124.csv'), 
         #                         index_col=0).loc[md.level_0]

	### NOW:
    clin_reduced=pd.read_csv(os.path.join(data_path,
                                  'clin_reduced_124.csv'), 
                                  index_col=0).loc[md.level_0]
    clin_full=pd.read_csv(os.path.join(data_path,
                                  'clin_full_124.csv'), 
                                  index_col=0).loc[md.level_0]
    
    clin = pd.concat([clin_reduced,clin_full[[#'weight_in_kg',
                                                     # 'mean_arterial_pressure',
                                                      'systolic_bp',
                                                      'diastolic_bp']
                                                      ]],axis=1)

    ## scale the luminex data
    ss=StandardScaler()
    
    luminex_processed = pd.concat([
        pd.DataFrame(ss.fit_transform(luminex.loc[md[md['batch_2']=='batch_1_2'].index]), index=md[md['batch_2']=='batch_1_2'].index,columns=luminex.columns).fillna(0),
        pd.DataFrame(ss.fit_transform(luminex.loc[md[md['batch_2']=='batch_3'].index]), index=md[md['batch_2']=='batch_3'].index,columns=luminex.columns).fillna(0),
    ]).loc[md.index]


     ## scale the clinical data
    ss=StandardScaler()
    clin_processed = pd.DataFrame( 
                          ss.fit_transform( clin ), 
                                    index=clin.index,
                                    columns=clin.columns
                                    ).fillna(0)

    clin_processed.index=md.index
    md['level_0']=md.index
    
    return(md, 
           micro_relabund, 
           luminex_processed, 
           clin_processed
           )
    

