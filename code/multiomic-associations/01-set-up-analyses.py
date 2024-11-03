import os 
CODE_DIR = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
    
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA
from deicode.rpca import rpca
from scipy.stats import pearsonr
import biom

from data_loading import load_data

import seaborn as sns
sns.set(rc={'axes.facecolor':'white', 
            'figure.facecolor':'white', 
            'axes.edgecolor':'black', 
            'grid.color': 'black'
            }, 
        style='ticks',
       font_scale=2)

def rescale(micro):
     return( micro/(micro.sum(axis=1)[:, np.newaxis] ) )

from skbio.stats.composition import clr
def mv_pseudocount(tbl):
    return np.power(10, np.floor(np.log10(rescale(tbl)[tbl>0].min().min())))

import warnings
warnings.filterwarnings('ignore')


def main(data_path=os.path.join(CODE_DIR,'../../data/'), 
         out_path=os.path.join(CODE_DIR,'../../results/multiomic-associations')
         ):
    
    md, micro_relabund, \
        luminex_processed, clin_processed = load_data(data_path=data_path)
    
    
    md.to_csv(os.path.join(out_path,
                           'diablo-setup', 
                           'metadata.csv'))
    
    luminex_processed.to_csv(os.path.join(out_path,
                             'diablo-setup', 
                             'luminex.csv'))
    
    clin_processed.to_csv(os.path.join(out_path,
                                       'diablo-setup', 
                                       'clinical.csv'))
    
    high_abund_cols=(micro_relabund>1e-3).mean(axis=0) > 0.05
    pd.DataFrame(clr(1e-8 + micro_relabund.loc[:, high_abund_cols].values ), 
                 index=micro_relabund.index, 
                 columns=micro_relabund.columns[high_abund_cols]
                ).to_csv(os.path.join(out_path,
                                      'diablo-setup', 
                                      'microbes.csv'))
    
    
    biom_tbl=biom.Table(micro_relabund.T.values,
                        observation_ids=micro_relabund.columns, 
                        sample_ids=md.index
                        )
    
    np.random.seed(123)
    n_comps=5
    ordination, distance = rpca(biom_tbl, 
                                n_components=n_comps, 
                                min_sample_count=0, 
                                min_feature_count=0, 
                                min_feature_frequency=0, 
                                )
    
    
    pc=PCA(n_components=2)
    xx=ordination.samples.PC1.values[:, np.newaxis]
    yy=pc.fit_transform(luminex_processed)
    lr=LinearRegression()
    plt.figure(figsize=(8,8))
    ax=sns.scatterplot(x=xx[:, 0], 
                    y=yy[:, 0], 
                    hue=md.PEC_case,
                    s=100
                    )
    sns.lineplot(x=xx[:, 0], 
                 y=lr.fit(xx[:, 0:1], yy[:, 0]).predict(xx[:, 0:1]), 
                 linewidth=5, 
                 color='black', 
                 ax=ax
                )
    plt.legend().remove()
    # plt.xticks([])
    # plt.yticks([])
    plt.xlabel('Microbe RPC1')
    plt.ylabel('Immune factor PC1')
    pear=pearsonr(xx[:, 0], yy[:, 0])
    plt.title('Pearson R: {:.2f}\np: {:.2e}'.format(*pear))
    plt.savefig('../../results/multiomic-associations/general-pc1-scatterplot.pdf',
                dpi=900, 
                bbox_inches='tight', 
                format='pdf')
    
    
    np.random.seed(123)
    n_comps=5
    ordination, distance = rpca(biom_tbl, 
                                n_components=n_comps, 
                                min_sample_count=0, 
                                min_feature_count=0, 
                                min_feature_frequency=0, 
                                )
    
    pc=PCA(n_components=3)
    yy=pc.fit_transform(luminex_processed)
    xx=ordination.samples.values
    
    qq=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            qq[i,j] = pearsonr(yy[:, i], xx[:, j])[0]
    
    pp=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            pp[i,j] = pearsonr(yy[:, i], xx[:, j])[1]


    print(qq)
    print(pp)
    
    
    plt.figure(figsize=(8,8))
    sns.heatmap(qq, 
                vmin=-.5, 
                vmax=.5, 
                cmap='coolwarm')
    plt.title('Pearson R of IF PCA vs Microbe RPCA')
    plt.ylabel('IF PC')
    plt.xlabel('Mic. RPC')
    plt.savefig('../../results/multiomic-associations/pc-corr-heatmap.pdf', 
                dpi=900, 
                format='pdf', 
                bbox_inches='tight'
                )

if __name__=='__main__':
    main()
        
                 





