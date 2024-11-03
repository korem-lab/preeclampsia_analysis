import os 
CODE_DIR = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV,LassoCV,LogisticRegression,LinearRegression
from sklearn.feature_selection import SelectKBest,SelectPercentile
from scipy.stats import mannwhitneyu
from skbio.stats.composition import clr
from sklearn.pipeline import Pipeline
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.stats.multitest import fdrcorrection
from scipy.spatial.distance import squareform, pdist
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.stats import fisher_exact

import sys
sys.path.append('../immune-factor-associations')
from run_IF_analysis import confidence_ellipse

import skbio
from skbio.tree import TreeNode

sns.set(rc={'axes.facecolor':'white', 
            'figure.facecolor':'white', 
            'axes.edgecolor':'black', 
            'grid.color': 'black',
            }, 
        style='ticks',
       font_scale=2)

def rescale(micro):
     return( micro/(micro.sum(axis=1).values[:, np.newaxis] ) )

def mv_pseudocount(tbl):
    return np.power(10, np.floor(np.log10(rescale(tbl)[tbl>0].min().min())))

def add_15_percent_to_top_ylim(ax):
    ### I'M ADJUSTING THIS FUNCTION TO MAKE IT 20 PERCENT TO YLIM, NOT 15
    ylim=ax.get_ylim() 
    ax.set_ylim( ylim[0], ylim[1] + ( ylim[1] - ylim[0] ) * .2)
    return(None)

import warnings
warnings.filterwarnings('ignore')


def load_data(data_path):
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
    md['PEC_32']=(md[md['PEC_any']==1]['GA_days_at_diagnosis']<(7*32)).astype(int)
    md.loc[md['PEC_any']==0,'PEC_32']=0

    md['PEC_34']=(md[md['PEC_any']==1]['GA_days_at_diagnosis']<(7*34)).astype(int)
    md.loc[md['PEC_any']==0,'PEC_34']=0

    micro=micro.loc[md.index]
    
    print(micro.shape)
    
    ## prevalence based filtering
    micro=micro.loc[:, ( rescale(micro) > 0).mean(axis=0) > 0.1]
    print(micro.shape)
    micro_relabund = micro/micro.values.sum(axis=1)[:, np.newaxis]
    
    luminex=pd.read_csv(os.path.join(data_path, 
                                     'luminex_lod_by_batch_124.csv'),
                        index_col=0).rename(index={v:k for k,v in md.level_0.to_dict().items()})

    luminex=luminex.loc[md.index]
    clin_reduced=pd.read_csv(os.path.join(data_path,
                                  'clin_reduced_124.csv'), 
                                  index_col=0).loc[md.level_0]
    clin_full=pd.read_csv(os.path.join(data_path,
                                  'clin_full_124.csv'), 
                                  index_col=0).loc[md.level_0]
    
    
    clin_reduced = pd.concat([clin_reduced,clin_full[['weight_in_kg','mean_arterial_pressure','systolic_bp','diastolic_bp']]],axis=1)

    
    ## scale the luminex data
    ss=StandardScaler()
    luminex_processed = pd.DataFrame( ss.fit_transform( luminex ), 
                                        index=luminex.index,
                                        columns=luminex.columns
                                        ).fillna(0)

     ## scale the clinical data
    ss=StandardScaler()
    clin_reduced_processed = pd.DataFrame( 
                          ss.fit_transform( clin_reduced ), 
                                    index=clin_reduced.index,
                                    columns=clin_reduced.columns
                                    ).fillna(0)

    clin_reduced.index=md.index
    
    clin_full_processed = pd.DataFrame( 
                          ss.fit_transform( clin_full ), 
                                    index=clin_full.index,
                                    columns=clin_full.columns
                                    ).fillna(0)

    clin_full_processed.index=md.index
    
    
    ## switching the VMCG adjustments back to standard
    micro_relabund.columns=\
                    micro_relabund.columns\
                        .str.replace('Bifidobacterium_vaginale', 'Gardnerella_vaginalis')\
                        .str.replace('Bifidobacterium_leopoldii', 'Gardnerella_leopoldii')\
                        .str.replace('Bifidobacterium_swidsinskii', 'Gardnerella_swidsinskii')\
                        .str.replace('Bifidobacterium_piotii', 'Gardnerella_piotii')

    return(md, 
           micro_relabund, 
           luminex_processed, 
           clin_reduced_processed,
           clin_full_processed
           )


def make_mannwhityneu_presence_volcano_plot(micro_,
                                            micro_clr,
                                            md,
                                            min_t=10/500000):
    vv=0#micro_.shape[0]*.01
    min_t = 0.3
    tmp_df = micro_.T
    
    inds=md.index
    tmp_df['grp'] = tmp_df.index.str[6:].str.rstrip('_H').str.rstrip('_F')\
                            .str.rstrip('_A').str.rstrip('_C').str.rstrip('_B')\
                            .str.replace('_unknown', '').str.replace(':', '')

    micro_spec = tmp_df.groupby('grp')[tmp_df.columns[:-1]].sum().T
    

    conds=lambda a: True
    
    smdf=pd.DataFrame(
        {'Gardnerella':micro_spec[[a for a in micro_spec.columns 
                        if 'Gardnerella' in a]].sum(axis=1), 
         'Lcrisp':micro_spec[[a for a in micro_spec.columns 
                        if 'Lactobacillus_crispatus' in a]].sum(axis=1), 
         'Liners':micro_spec[[a for a in micro_spec.columns 
                        if 'Lactobacillus_iners' in a]].sum(axis=1)}
        )

    smdf = ( smdf > 0.3 ) * (smdf == smdf.max(axis=1).values[:, np.newaxis] )
    
    mws = [ fisher_exact(
            pd.crosstab(smdf.loc[inds][a], 
                        md.loc[inds]['PEC_case']) 
                )[1]
        for a in smdf.columns[:3] ] + \
    [ fisher_exact(
            pd.crosstab(smdf.max(axis=1)==False, 
                        md.loc[inds]['PEC_case']) 
                )[1] ]


    mw_test = pd.DataFrame({'ps':mws, 
                 'otu':['Gardnerella',
                        'Lactobacillus crispatus',
                        'Lactobacillus iners',
                        'None',
                        ], 
                 'fdr':fdrcorrection(mws, alpha=.2)[1]
                 })\
        .sort_values('ps')
    
    mw_test['effect_size'] = np.log10( [ fisher_exact(
                                    pd.crosstab(smdf.loc[inds][a], 
                                                md.loc[inds]['PEC_case']) 
                                        )[0]
                                for a in smdf.columns[:3] ] + \
                            [ fisher_exact(
                                    pd.crosstab(smdf.max(axis=1)==False, 
                                                md.loc[inds]['PEC_case']) 
                                        )[0] ] )


    mw_test['negativelog10p']=-np.log10(mw_test['ps'])


    plt.figure(figsize=(8,8))
    ax=sns.scatterplot(x='effect_size', 
                       y='negativelog10p',
    #                 y='da_analysis$p',
                    data=mw_test, 
                       hue= mw_test.otu.str.lower().str.contains('gardnerella')*3 +\
                       mw_test.otu.str.lower().str.contains('bifidobacterium')*2 + \
                               mw_test.otu.str.lower().str.contains('prevotella')*1, #_timonensis
                       palette = {0:'black',
                                  1:'#C178EE',
                                  2:'#71F030',
                                  3:'#E97451',
                                  },
                       s=100,
#                        color='black'
                    )
    
    print(mw_test.fdr.min())
    
    tts=( mw_test.loc[mw_test['fdr']<.2]['negativelog10p'].min() +\
            mw_test.loc[mw_test['fdr']>.2]['negativelog10p'].max() )/2
    

    print('PRESENCE-ABSENCE:')

    print(mw_test.head(25))

#     print(tts)

    qqq=ax.get_xlim()
    axxlim=ax.get_xlim()
    
    
    ax=sns.lineplot(x=[axxlim[0]-10,
                       axxlim[1]+10], 
                 y=np.array([tts, 
                             tts
                            ]),
                 color='red', 
                 linewidth=5, 
                 linestyle='--',
                 ax=ax, 
                 ci=0
                 )

    tts=( mw_test.loc[mw_test['fdr']<.1]['negativelog10p'].min() +\
        mw_test.loc[mw_test['fdr']>.1]['negativelog10p'].max() )/2
    print(mw_test.head())

    print(tts)

    ax=sns.lineplot(x=[axxlim[0]-10,
                       axxlim[1]+10], 
                 y=np.array([tts, 
                             tts
                            ]),
                 color='red', 
                 linewidth=5, 
                 linestyle='--',
                 ax=ax, 
                 ci=0
                 )
#     except:
#         pass
    plt.xlim(axxlim)
    plt.ylim([-0.1,3.1])
    plt.legend().remove()
    
    return(ax, mw_test, smdf)
    



def make_mannwhityneu_volcano_plot(micro,
                                   micro_clr,
                                   md,
                                   min_t=10/500000):
    conds = lambda a: (micro[a][md.PEC_case] > min_t).sum() >=10 and \
                         (micro[a][~md.PEC_case] > min_t).sum() >=10


    mws = [ mannwhitneyu(micro_clr[a].values[md.PEC_case][
                                micro[a].values[md.PEC_case] > min_t ],
                         micro_clr[a].values[~md.PEC_case][
                                 micro[a].values[~md.PEC_case] > min_t],
                         alternative='two-sided'
                         ).pvalue
           for a in micro.columns[:-1]
           if conds(a)
          ]

    mw_test = pd.DataFrame({'ps':mws, 
                 'otu':[a for a in micro.columns[:-1] 
                        if conds(a)
                        ], 
                 'fdr':fdrcorrection(mws, alpha=.2)[1]
                 })\
        .sort_values('ps')
    mw_test['effect_size'] = [ ( micro_clr.loc[:, val][md.PEC_case][
                                micro[val].values[md.PEC_case] > min_t ] ).mean() -\
                                ( micro_clr.loc[:, val][~md.PEC_case][
                                micro[val].values[~md.PEC_case] > min_t ]).mean()
                             for val in mw_test.otu.values]



    mw_test['negativelog10p']=-np.log10(mw_test['ps'])


    plt.figure(figsize=(8,8))
    ax=sns.scatterplot(x='effect_size', 
                       y='negativelog10p',
                    data=mw_test, 
                       hue= mw_test.otu.str.lower().str.contains('gardnerella')*3 +\
                       mw_test.otu.str.lower().str.contains('bifidobacterium')*2 + \
                               mw_test.otu.str.lower().str.contains('prevotella')*1, #_timonensis
                       palette = {0:'black',
                                  1:'#C178EE',
                                  2:'#71F030',
                                  3:'#E97451',
                                  },
                       s=100,
                    )
    
    print(mw_test.fdr.min())
    
    tts=( mw_test.loc[mw_test['fdr']<.2]['negativelog10p'].min() +\
            mw_test.loc[mw_test['fdr']>.2]['negativelog10p'].max() )/2
    


    print(mw_test.head())

    print(tts)

    qqq=ax.get_xlim()
    axxlim=ax.get_xlim()
    
    
    ax=sns.lineplot(x=[axxlim[0]-10,
                       axxlim[1]+10], 
                 y=np.array([tts, 
                             tts
                            ]),
                 color='red', 
                 linewidth=5, 
                 linestyle='--',
                 ax=ax, 
                 ci=0
                 )

    tts=( mw_test.loc[mw_test['fdr']<.1]['negativelog10p'].min() +\
        mw_test.loc[mw_test['fdr']>.1]['negativelog10p'].max() )/2
    

    print(mw_test.head())

    print(tts)

    ax=sns.lineplot(x=[axxlim[0]-10,
                       axxlim[1]+10], 
                 y=np.array([tts, 
                             tts
                            ]),
                 color='red', 
                 linewidth=5, 
                 linestyle='--',
                 ax=ax, 
                 ci=0
                 )
    
    plt.xlim(axxlim)
    
    if mw_test.negativelog10p.max() <= 3.1:
        plt.ylim([-0.1,3.1])
    plt.legend().remove()
    
    return(ax, mw_test)
    
    



def main(data_path=os.path.join(CODE_DIR,'../../data/'),
         out_path=os.path.join(CODE_DIR,'../../results/microbe-associations'),
         use_confidence_ellipse=True,
         seed=1
         ):

    np.random.seed(seed)
    md, micro, luminex, clin_reduced, clin_full = load_data(data_path=data_path)
    
    micro_clr = pd.DataFrame(clr(mv_pseudocount(micro)+micro), 
                             index=micro.index, 
                             columns=micro.columns
                             )
    y=md.PEC_case.values
    md['BMIg25'] = md['BMI']>=25
    md['BMIl25'] = md['BMI']<25
    
    ax, mw_test = make_mannwhityneu_volcano_plot(micro,
                                                 micro_clr,
                                                 md)
    mw_test.to_csv("microbe_associations_full_cohort.csv")
    
    plt.savefig(os.path.join(out_path,
                             'volcano_full.pdf'),
                format='pdf', 
                bbox_inches='tight', 
                dpi=900
               )
    
    all_f_teas=[]
    ax, fisher_teas, smdf = make_mannwhityneu_presence_volcano_plot(micro,
                                                 micro_clr,
                                                 md)
    plt.savefig(os.path.join(out_path,
                             'volcano_presence_full.pdf'),
                format='pdf', 
                bbox_inches='tight', 
                dpi=900
               )
    
    smdf['None'] = ( 1 - smdf.sum(axis=1) ).astype(bool)
    qqq1 = pd.concat([smdf.idxmax(axis=1), 
               md.loc[smdf.index][['PEC_case']]], axis=1)
    
    qqq2= ( qqq1.loc[qqq1.PEC_case][0].value_counts() / qqq1.PEC_case.sum() 
                  ).reset_index()
    qqq2['PEC_case']=True
    
    qqq3= ( qqq1.loc[~qqq1.PEC_case][0].value_counts() / (qqq1.PEC_case==False).sum()
                     ).reset_index()
    qqq3['PEC_case']=False
    
    qqq1 = pd.concat([qqq2, qqq3])
    qqq1.columns=['CST', 'Perc', 'PEC_case']
    
    plt.figure(figsize=(8,8))
    sns.barplot(x='CST', 
                y='Perc', 
                hue='PEC_case', 
                data=qqq1, 
                order=['Lcrisp', 'Liners', 'Gardnerella', 'None'],
                hue_order=[False, True],
                palette={True:'gold', False:'lightblue'}
               )
    plt.legend().remove()
    plt.savefig(os.path.join(out_path,
                                'presence-barplot-full.pdf'),
                    format='pdf', 
                    bbox_inches='tight', 
                    dpi=900
                   )
    
    
    fisher_teas['group']='full'
    all_f_teas += [fisher_teas.copy()]
    
    min_t = 10./500000
    
    ## make the boxplots
    for sps in mw_test.otu.head(3).values:
        print(sps)
        
        
        inds=md.BMIg25.isin([True, False]).loc[micro.loc[:, sps]>min_t]
        print('all')
        print(
               mannwhitneyu( 
           micro_clr.loc[:, sps][micro.loc[:, sps]>min_t].loc[inds]\
                 .loc[md.loc[inds.index].loc[inds].PEC_case], 
                 micro_clr.loc[:, sps][micro.loc[:, sps]>min_t]
                   .loc[inds]\
              .loc[~md.loc[inds.index].loc[inds].PEC_case], 
                   alternative='two-sided'
                           ).pvalue )
        
        inds=md.BMIg25.loc[micro.loc[:, sps]>min_t]
        print('BMIg25')
        print(
               mannwhitneyu( 
           micro_clr.loc[:, sps][micro.loc[:, sps]>min_t].loc[inds]\
                 .loc[md.loc[inds.index].loc[inds].PEC_case], 
                 micro_clr.loc[:, sps][micro.loc[:, sps]>min_t]
                   .loc[inds]\
              .loc[~md.loc[inds.index].loc[inds].PEC_case], 
                   alternative='two-sided'
                           ).pvalue )
        
        print('BMIl25')
        inds=md.BMIl25.loc[micro.loc[:, sps]>min_t]
        print(
               mannwhitneyu( 
           micro_clr.loc[:, sps][micro.loc[:, sps]>min_t].loc[inds]\
                 .loc[md.loc[inds.index].loc[inds].PEC_case], 
                 micro_clr.loc[:, sps][micro.loc[:, sps]>min_t]
                   .loc[inds]\
              .loc[~md.loc[inds.index].loc[inds].PEC_case], 
                   alternative='two-sided'
                           ).pvalue )
        
        plt.figure(figsize=(4,8))
        ax=sns.boxplot( y= micro_clr.loc[:, sps][micro.loc[:, sps]>min_t], 
                        hue= md.PEC_case[micro.loc[:, sps]>min_t],
                        x=md.BMIg25[micro.loc[:, sps]>min_t],
                        hue_order=[False, True],
                        order=[False, True],
                        palette={True:'gold', False:'lightblue'}
                     )

        sns.swarmplot(y= micro_clr.loc[:, sps][micro.loc[:, sps]>min_t], 
                      hue= md.PEC_case[micro.loc[:, sps]>min_t], 
                      x=md.BMIg25[micro.loc[:, sps]>min_t],
                      hue_order=[False, True],
                      order=[False, True],
                      s=10, 
                      color='black',
                      ax=ax, 
                      dodge=True
                     )
        ax.legend().remove()
        add_15_percent_to_top_ylim(ax)
        
        plt.savefig(os.path.join(out_path,
                                 sps+'boxplot.pdf'),
                    format='pdf', 
                    bbox_inches='tight', 
                    dpi=900
                   )

    ax, mw_test = make_mannwhityneu_volcano_plot(micro.loc[md.BMIl25],
                                                 micro_clr.loc[md.BMIl25],
                                                 md.loc[md.BMIl25])
    mw_test.to_csv("microbe_associations_BMI_under_25.csv")
    
    plt.savefig(os.path.join(out_path,
                             'volcano_bmil25.pdf'),
                format='pdf', 
                bbox_inches='tight', 
                dpi=900
               )
    
    ax, fisher_teas, smdf = make_mannwhityneu_presence_volcano_plot(micro.loc[md.BMIl25],
                                                         micro_clr.loc[md.BMIl25],
                                                         md.loc[md.BMIl25])
    
    fisher_teas['group']='bmil25'
    all_f_teas += [fisher_teas.copy()]
    
    plt.savefig(os.path.join(out_path,
                             'volcano_presence_bmil25.pdf'),
                format='pdf', 
                bbox_inches='tight', 
                dpi=900
               )
    
    smdf['None'] = ( 1 - smdf.sum(axis=1) ).astype(bool)
    qqq1 = pd.concat([smdf.idxmax(axis=1), 
               md.loc[smdf.index][['PEC_case']]], axis=1)
    
    qqq2= ( qqq1.loc[qqq1.PEC_case][0].value_counts() / qqq1.PEC_case.sum() 
                  ).reset_index()
    qqq2['PEC_case']=True
    
    qqq3= ( qqq1.loc[~qqq1.PEC_case][0].value_counts() / (qqq1.PEC_case==False).sum()
                     ).reset_index()
    qqq3['PEC_case']=False
    
    qqq1 = pd.concat([qqq2, qqq3])
    qqq1.columns=['CST', 'Perc', 'PEC_case']
    
    plt.figure(figsize=(8,8))
    sns.barplot(x='CST', 
                y='Perc', 
                hue='PEC_case', 
                data=qqq1, 
                order=['Lcrisp', 'Liners', 'Gardnerella', 'None'],
                hue_order=[False, True],
                palette={True:'gold', False:'lightblue'}
               )
    plt.legend().remove()
    plt.savefig(os.path.join(out_path,
                                'presence-barplot-bmil25.pdf'),
                    format='pdf', 
                    bbox_inches='tight', 
                    dpi=900
                   )
    
    
    ax, mw_test = make_mannwhityneu_volcano_plot(micro.loc[md.BMIg25],
                                                 micro_clr.loc[md.BMIg25],
                                                 md.loc[md.BMIg25])
    mw_test.to_csv("microbe_associations_BMI_above_25.csv")
    
    print((mw_test.fdr<0.2).sum())
    
    
    plt.savefig(os.path.join(out_path,
                             'volcano_bmig25.pdf'),
                format='pdf', 
                bbox_inches='tight', 
                dpi=900
               )
    
    ax, fisher_teas, smdf = make_mannwhityneu_presence_volcano_plot(micro.loc[md.BMIg25],
                                                         micro_clr.loc[md.BMIg25],
                                                         md.loc[md.BMIg25])
    
    fisher_teas['group']='bmig25'
    all_f_teas += [fisher_teas.copy()]
    
    plt.savefig(os.path.join(out_path,
                             'volcano_presence_bmig25.pdf'),
                format='pdf', 
                bbox_inches='tight', 
                dpi=900
               )
    
    
    smdf['None'] = ( 1 - smdf.sum(axis=1) ).astype(bool)
    qqq1 = pd.concat([smdf.idxmax(axis=1), 
               md.loc[smdf.index][['PEC_case']]], axis=1)
    
    qqq2= ( qqq1.loc[qqq1.PEC_case][0].value_counts() / qqq1.PEC_case.sum() 
                  ).reset_index()
    qqq2['PEC_case']=True
    
    qqq3= ( qqq1.loc[~qqq1.PEC_case][0].value_counts() / (qqq1.PEC_case==False).sum()
                     ).reset_index()
    qqq3['PEC_case']=False
    
    qqq1 = pd.concat([qqq2, qqq3])
    qqq1.columns=['CST', 'Perc', 'PEC_case']
    
    plt.figure(figsize=(8,8))
    sns.barplot(x='CST', 
                y='Perc', 
                hue='PEC_case', 
                data=qqq1, 
                order=['Lcrisp', 'Liners', 'Gardnerella', 'None'],
                hue_order=[False, True],
                palette={True:'gold', False:'lightblue'}
               )
    plt.legend().remove()
    plt.savefig(os.path.join(out_path,
                                'presence-barplot-bmig25.pdf'),
                    format='pdf', 
                    bbox_inches='tight', 
                    dpi=900
                   )
    
    
    pd.concat(all_f_teas).to_csv( os.path.join(out_path,
                                               'all_presence_fishers.csv' ) )
    
    ### PCA scatterplot
#     pca = PCA(n_components=2)
#     X = pca.fit_transform(micro_clr)

    from deicode.rpca import rpca
    import biom
    
    ordination, distance = rpca(biom.Table(micro.values.T,
                                            observation_ids=micro.columns, 
                                            sample_ids=md.index
                                            ), 
                                 n_components=2, 
                                 min_sample_count=0, 
                                 min_feature_count=0, 
                                 min_feature_frequency=0
                               )
    pca = ordination
    X = ordination.samples.values
    
    
    print('all permanova')
    print(permanova(distance, md['PEC_case'], permutations=9999))
    
    
    inds=~md.BMIg25
    ordination, distance = rpca(biom.Table(micro.loc[inds].values.T,
                               observation_ids=micro.columns, 
                               sample_ids=md.loc[inds].index
                               ), 
                             n_components=2, 
                             min_sample_count=0, 
                             min_feature_count=0, 
                             min_feature_frequency=0
                               )
    
    print('BMI < 25 permanova')
    print(permanova(distance, md.loc[inds]['PEC_case'], permutations=9999))
    
    
    inds=md.BMIg25
    ordination, distance = rpca(biom.Table(micro.loc[inds].values.T,
                               observation_ids=micro.columns, 
                               sample_ids=md.loc[inds].index
                               ), 
                         n_components=2, 
                         min_sample_count=0, 
                         min_feature_count=0, 
                         min_feature_frequency=0
                               )
    
    print('BMI > 25 permanova')
    print(permanova(distance, md.loc[inds]['PEC_case'], permutations=9999))
#     temp=ordination
    
        
    
    fig = plt.figure(figsize = (8,8))
    kwargs = {'x': X[:,0], 'y':X[:,1]}

    sns.scatterplot(**kwargs,
                    s=100, 
                    ax = plt.subplot(111), 
                    hue=md.PEC_case,
                    palette={True:'gold', False:'lightblue'}
                    )
#     plt.xlabel("PC1, EV = {:.0f}%".format(pca.explained_variance_ratio_[0]*100))
#     plt.ylabel("PC2, EV = {:.0f}%".format(pca.explained_variance_ratio_[1]*100))
    plt.xlabel("PC1, EV = {:.0f}%".format(pca.proportion_explained[0]*100))
    plt.ylabel("PC2, EV = {:.0f}%".format(pca.proportion_explained[1]*100))
    plt.legend(bbox_to_anchor=(1.45,1))


    
    if use_confidence_ellipse:
        temp = pd.DataFrame(X, index=md.index).loc[inds]

        temp['PEC_case'] = md.loc[inds]['PEC_case']

        x1 = temp[temp['PEC_case'] == True][0]
        y1 = temp[temp['PEC_case'] == True][1]

        confidence_ellipse(x1, y1, plt.gca(), edgecolor='goldenrod', n_std=2)

        x2 = temp[temp['PEC_case'] == False][0]
        y2 = temp[temp['PEC_case'] == False][1]

        confidence_ellipse(x2, y2, plt.gca(), edgecolor='slateblue', n_std=2)
        
        plt.savefig(os.path.join(out_path, 'PCA_scatteplot.pdf'),
                format='pdf',
                dpi=900, 
                bbox_inches='tight'
                )
    else:
        plt.savefig(os.path.join(out_path, 'PCA_scatteplot-no-confidence-ellipse.pdf'),
                format='pdf',
                dpi=900, 
                bbox_inches='tight'
                )
    
    
    
    
    ### pca plots
    inds=md.index
    ww=X
    print(ww)
    
    print('all')
    inds=md.BMIg25.isin([True, False])
    mann= mannwhitneyu(ww[inds][:, 0][ md.loc[inds].PEC_case ], 
                       ww[inds][:, 0][~md.loc[inds].PEC_case ],
                       alternative='two-sided'
                       ).pvalue
    print(mann)
    print('bmig25')
    
    
    inds=md.BMIg25
    mann= mannwhitneyu(ww[inds][:, 0][ md.loc[inds].PEC_case ], 
                       ww[inds][:, 0][~md.loc[inds].PEC_case ],
                       alternative='two-sided'
                       ).pvalue
    print('bmig25')
    print(mann)
    print('bmil25')
    print( mannwhitneyu(ww[~inds][:, 0][ md.loc[~inds].PEC_case ], 
                        ww[~inds][:, 0][ ~md.loc[~inds].PEC_case ],
                        alternative='two-sided'
                        ).pvalue )
    
    inds=md.index
    plt.figure(figsize=(4,8))
    ax=sns.boxplot(y=ww[:, 0],
                   hue=md.loc[inds].PEC_case,
                   x=md.loc[inds].BMIg25,
                   hue_order=[False, True],
                   order=[False, True],
                       palette={True:'gold', False:'lightblue'}
                   )
    sns.swarmplot(y=ww[:, 0],
                  hue=md.loc[inds].PEC_case, 
                  color='black',
                  x=md.loc[inds].BMIg25,
                  hue_order=[False, True],
                  order=[False, True],
                  dodge=True,
                  s=10, 
                  ax=ax
                      )
        
    ax.legend().remove()
    
    add_15_percent_to_top_ylim(ax)
    plt.title('PCA of micro clr' + \
              '\nMannwhitneyu' + \
              ' p: {:.2f}'.format(mann)
             )

    plt.savefig(os.path.join(out_path,
                             'All-PCA-boxplot.pdf'),
                format='pdf', 
                bbox_inches='tight', 
                dpi=900
                )
    
    
if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    