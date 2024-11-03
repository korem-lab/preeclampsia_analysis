import os 
CODE_DIR = os.getcwd()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV,LassoCV,LogisticRegression
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.model_selection import LeaveOneOut,StratifiedKFold,KFold,RepeatedStratifiedKFold,RepeatedKFold,GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, \
                                precision_recall_curve, average_precision_score
from skbio.stats.composition import clr
from scipy.stats import zscore 
from sklearn.pipeline import Pipeline
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid, cross_val_predict, cross_val_score
from rebalancedcv import RebalancedLeaveOneOut

import torch
import torch.nn.functional as F
torch.manual_seed(1)
np.random.seed(1)


from debiasm.sklearn_functions import DebiasMClassifierLogAdd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


sns.set(rc={'axes.facecolor':'white', 
            'figure.facecolor':'white', 
            'axes.edgecolor':'black', 
            'grid.color': 'black'
            }, 
       font_scale=2)
pal = sns.color_palette()

def format_aupr_dfs(a, name_dict_pred_map, y_test):
    prds=name_dict_pred_map[a]
    pre, rec, __ = precision_recall_curve(y_test, 
                                          prds
                                          )
    area = auc(rec, pre)
    rec -= pre*1e-7 ## small shift to make PR curve point order correct in seaborn
    return( pd.DataFrame({'Precision':pre, 
                          'Recall':rec, 
                          'Group':a + ' (auPR = {:.2f})'.format(area)}
                        )
          )

def format_auroc_dfs(a, name_dict_pred_map, y_test):
    prds = name_dict_pred_map[a]
    fpr, tpr, __ = roc_curve(y_test, 
                             prds, 
                             drop_intermediate=False
                             )
    area = auc(fpr, tpr)
    fpr = fpr + tpr*1e-7
    return( pd.DataFrame({'FPR':fpr, 
                          'TPR':tpr, 
                          'Group':a + ' (auROC = {:.2f})'.format(area)}
                        )
          )

normalize=lambda x: StandardScaler().fit_transform(x[:, np.newaxis])[:, 0]

def batch_weight_feature_and_nbatchpairs_scaling(strength, df_with_batch):
    nbs = df_with_batch.iloc[:, 0].nunique()
    w = nbs * (nbs-1) / 2
    return(strength /( w * ( df_with_batch.shape[1] - 1 ) ) )

def rescale(micro):
     return( micro/(micro.sum(axis=1).values[:, np.newaxis] ) )

def mv_pseudocount(tbl):
    return np.power(10, np.floor(np.log10(rescale(tbl)[tbl>0].min().min())))

import warnings
warnings.filterwarnings('ignore')



def run_multilevel_cv_tuning(X,
                             y,
                             md, 
                             seed,
                             rloocv=RebalancedLeaveOneOut()
                             ):
    
    np.random.seed(seed)
    # Note- same pipeline as internal NuMoM cross-validation 
    pipe = Pipeline([
            ('skb', SelectKBest()),
            ('lrg', LogisticRegression())
        ])
    params = {
        'skb__k': [2, 5, 10],
        'lrg__C': np.logspace(0, 8, 10), 
    }
    
    if X.shape[1]<10:
        params = {
                  'skb__k': [2,5],
                  'lrg__C': np.logspace(0, 8, 10), 
                  }
    
        
    full_preds = []
    bmig25_preds = []
    bmigl5_preds = []
    all_params = []

    
    # Tune hyperparams on NuMom samples ONLY using rebalanced-leave-one-out cross validation
    best_params = None
    best_score = -np.inf
    for i,perm in enumerate(ParameterGrid(params)):
        pipe.set_params(**perm)

        all_params.append(perm)

        full_preds.append( cross_val_predict(pipe,
                                             X,
                                             y,
                                             groups=md['BMIg25'],
                                             cv=rloocv,
                                             method='predict_proba',
                                              ) )

        bmig25_preds.append( cross_val_predict(pipe,
                                               X[md['BMIg25']],
                                               y[md['BMIg25']],
                                               method='predict_proba',
                                              cv=rloocv)) 

        bmigl5_preds.append( cross_val_predict(pipe,
                                               X[~md['BMIg25']],
                                               y[~md['BMIg25']],
                                                method='predict_proba',
                                      cv=rloocv) 
                                      ) #)

    best_model_g25 = clone(pipe).set_params(
        **all_params[np.argmax( [ roc_auc_score( y[md['BMIg25']], a[:, 1] )
                                 for a in bmig25_preds ] )])
    best_model_g25.fit(X[md['BMIg25']],y[md['BMIg25']])

    best_model_l25 = clone(pipe).set_params(
        **all_params[np.argmax( [ roc_auc_score( y[~md['BMIg25']], a[:, 1] ) 
                                 for a in bmigl5_preds ] )])

    best_model_l25.fit(X[~md['BMIg25']],y[~md['BMIg25']])
    
    best_model_overall = clone(pipe).set_params(
        **all_params[np.argmax( [ roc_auc_score( y, a[:, 1] ) 
                                 for a in full_preds ] )])
    
    best_model_overall.fit(X, y)
    
    
    use_bmi_split_g25 = max( ( [ roc_auc_score( y[md['BMIg25']], a[:, 1] )
                                 for a in bmig25_preds ] ) ) > \
                    max( ( [ roc_auc_score( y[md['BMIg25']], a[md['BMIg25'] ][:, 1] )
                                     for a in full_preds ] ) )


    use_bmi_split_l25 = max( ( [ roc_auc_score( y[~md['BMIg25']], a[:, 1] )
                            for a in bmigl5_preds ] ) ) > \
                        max( ( [ roc_auc_score(y[~md['BMIg25']], 
                           a[~md['BMIg25'] ][:, 1] )
                                     for a in full_preds ] ) )
    
    return(best_model_overall, 
           best_model_g25,
           best_model_l25,
           use_bmi_split_g25,
           use_bmi_split_l25)


def load_data(data_path):
    
    micro=pd.read_csv(
                os.path.join(data_path, 
                             'numom_hgf_kraken2_VMGC_species_abun_sub500K.csv'
                            ), 
                      index_col=0
                     )

    rcs=pd.read_csv(os.path.join(data_path, 
                                 'numom_hgf_PEC_subsampled_read_counts.csv'
                                ), 
                index_col=0)
    micro=micro.loc[micro.index.isin(rcs.index)]
    rcs=rcs.loc[micro.index]

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

    return(md, 
           micro_relabund, 
           luminex_processed, 
           clin_reduced_processed,
           clin_full_processed, 
           ss
           )

def transform(self, X):
    if self.model is None:
        raise(ValueError('You must run the `.fit()` method before executing this transformation'))

    x = torch.tensor(X.values)
    batch_inds, x = x[:, 0], x[:, 1:]
    x = F.normalize( torch.pow(2, self.model.batch_weights[batch_inds.long()] ) * x, p=1 )
    return( pd.DataFrame(x.detach().numpy(), 
                         index=X.index, 
                         columns=X.columns[1:]))




def main(data_path=os.path.join(CODE_DIR,'../../data/'),
         results_path=os.path.join(CODE_DIR,'../../results/predictions/'),
         seed=1
        ):

    ## the data path is in the git repo
    md, micro, luminex, clin_reduced, clin_full, ss = load_data(data_path=data_path)
    micro_clr = pd.DataFrame(clr(mv_pseudocount(micro)+micro), 
                         index=micro.index, 
                         columns=micro.columns
                         )
    y=md.PEC_case.values
    md['BMIg25'] = md['BMI']>=25

    momspi_md = pd.read_csv(os.path.join(data_path, 
                                     'momspi_md_16weeks.csv'),
                        index_col=0)

    momspi_md.loc[momspi_md['complications_eclampsia']=='Yes','PEC']=True
    momspi_md.loc[momspi_md['complications_eclampsia']=='No','PEC']=False

    momspi = pd.read_csv(os.path.join(data_path, 
                                  'momspi_kraken2_VMGC_species_abun_sub500K.csv'
                                  ),
                     index_col=0)


    momspi['unmapped'] = momspi_md['num_reads'] - momspi.sum(axis=1)
    
    momspi = rescale( momspi.loc[:,micro_clr.columns] )
    
    momspi_md = momspi_md.sort_values(['weeks_pregnant'])\
                        .groupby('subject').head(1)

    momspi_md=momspi_md.loc[momspi_md.num_reads>=500000]

    momspi_clr = pd.DataFrame(clr(mv_pseudocount(micro)+momspi), 
                         index=momspi.index, 
                         columns=momspi.columns
                         ).loc[momspi_md.index]
    
    X_test = momspi_clr.loc[momspi_md.index]

    micro_clr = pd.DataFrame(clr(mv_pseudocount(micro)+micro), 
                         index=micro.index, 
                         columns=micro.columns
                         )


    ## set up DEBIAS-M inputs
    momspi_clr['batch']=1
    momspi_clr = momspi_clr[['batch'] + list(momspi_clr.columns[:-1])]

    micro_clr['batch']=0
    micro_clr = micro_clr[['batch'] + list(micro_clr.columns[:-1])]

    torch.manual_seed(seed)
    np.random.seed(seed)

    dmc = DebiasMClassifierLogAdd(x_val=momspi_clr.values, 
                                batch_str=batch_weight_feature_and_nbatchpairs_scaling(1e3, 
                                                     pd.concat([momspi_clr, 
                                                                micro_clr])
                                                    ),
                               )

    dmc.fit(micro_clr.values, y)

    micro_clr=dmc.transform(micro_clr)
    momspi_clr=dmc.transform(momspi_clr)    
    
    ## run luminex training and predictions
    momspi_lum = pd.read_csv('../../data/momspi_cytokines_processed.csv', index_col=0)
    momspi_lum['subject']=momspi_lum.index.str.split('_').str[0]
    momspi_lum=momspi_lum.sort_index().groupby('subject').tail(1)\
                .set_index('subject').loc[momspi_md.subject]
    momspi_lum.index=momspi_md.index
    
    print(momspi_lum.shape)
    momspi_lum = momspi_lum.loc[:, momspi_lum.nunique() > 1 ]
    
    
    momspi_lum = pd.DataFrame( ss.fit_transform( momspi_lum ), 
                                index=momspi_lum.index,
                                columns=momspi_lum.columns
                                )
    
    print(momspi_lum.shape)
    luminex=luminex.loc[:, momspi_lum.columns]
    
    ## clinical inputs
    
    clin_cols = ['age', 
                 'bmi', 
                 'weight_in_kg', 
                 'systolic', 
                 'diastolic', 
                 'mean_arterial_pressure'
                 ]

    momspi_diastol_pretransform = pd.read_csv(os.path.join(data_path, 
                                                          'momspi_md_for_prediction.csv'
                                                       ),index_col=0
                                             ).drop_duplicates(subset='Run')\
                                            .set_index('Run')[clin_cols]\
                                                .loc[X_test.index]
    
    momspi_diastol = pd.DataFrame( 
                        StandardScaler().fit_transform(momspi_diastol_pretransform), 
                         index=momspi_diastol_pretransform.index,
                          columns=momspi_diastol_pretransform.columns )

    X_clin = clin_full[['Age_at_V1', 
                        'BMI',
                        'weight_in_kg', 
                        'systolic_bp', 
                        'diastolic_bp', 
                        'mean_arterial_pressure'
                         ]]
    
    X_clin = pd.DataFrame( 
                        StandardScaler().fit_transform(X_clin), 
                         index=X_clin.index,
                          columns=X_clin.columns )
    
    
    name_dfs_dict = {'Microbiome': [micro_clr, momspi_clr],
                     'Immune Factors': [luminex, momspi_lum],
                     'Clinical':[X_clin, momspi_diastol]}
    
    name_dict_pred_map = { }
    
    temp = momspi_diastol_pretransform[['age']]
    
    for a in name_dfs_dict:
        print(a)
        
        best_model_overall, \
           best_model_g25,\
           best_model_l25,\
           use_bmi_split_g25,\
           use_bmi_split_l25 = run_multilevel_cv_tuning(name_dfs_dict[a][0],
                                                        y,
                                                        md, 
                                                        seed=seed,
                                                        rloocv=RebalancedLeaveOneOut()
                                                        )
        
        print(use_bmi_split_g25, use_bmi_split_l25)
        
        momspi_g25 = momspi_diastol_pretransform['bmi'] >= 25
        momspi_l25 = momspi_diastol_pretransform['bmi'] < 25
        
        pred_col_nm='{}_pred'.format(a)
        temp[pred_col_nm] = np.nan    
        
        temp.loc[momspi_g25, pred_col_nm] = \
                            [ best_model_g25.predict_proba(name_dfs_dict[a][1][momspi_g25])[:,1]
                              if use_bmi_split_g25 else
                       best_model_overall.predict_proba(name_dfs_dict[a][1][momspi_g25])[:,1] ][0]
        
        temp.loc[momspi_l25, pred_col_nm] = \
                            [ best_model_l25.predict_proba(name_dfs_dict[a][1][~momspi_g25])[:,1]
                              if use_bmi_split_g25 else
                     best_model_overall.predict_proba(name_dfs_dict[a][1][~momspi_g25])[:,1] ][0]

        name_dict_pred_map[a] = temp[pred_col_nm].values
        
        
    y_test = momspi_md.loc[X_test.index].PEC.astype(int).values
        
    name_dict_pred_map[ 'Microbiome and Immune' ] = \
                normalize( name_dict_pred_map['Microbiome'] ) + \
                  normalize( name_dict_pred_map['Immune Factors'] )
        
        
    name_dict_pred_map[ 'Combined' ] = \
                normalize( name_dict_pred_map['Microbiome'] ) + \
                  normalize( name_dict_pred_map['Immune Factors'] ) + \
                    normalize( name_dict_pred_map['Clinical'] )
    
    
    print({a: roc_auc_score(y_test, name_dict_pred_map[a]) for a in name_dict_pred_map})
    print({a: average_precision_score(y_test, name_dict_pred_map[a]) for a in name_dict_pred_map})
    
    import itertools
    import delong
    res=pd.DataFrame( name_dict_pred_map ) 
    print( pd.DataFrame( [(a,
      b,
      np.power(10,
             delong.delong_roc_test(y_test, 
                                    res[a], 
                                    res[b]
                                    ) )[0][0] )
      for a,b in 
      itertools.combinations(res.columns, 2)], 
                 columns=['M1', 'M2', 'Delong P']) )

    plot_df = pd.concat([format_auroc_dfs(a, name_dict_pred_map, y_test) 
                         for a in name_dict_pred_map]
                        ).reset_index(drop=True)

    ### slight shifts to make all lines visible
    plot_df.loc[plot_df.Group.str.contains('Microbiome'), 'TPR'] = \
        plot_df.loc[plot_df.Group.str.contains('Microbiome'), 'TPR']+.5e-2

    plot_df.loc[plot_df.Group.str.contains('Microbiome'), 'FPR'] = \
        plot_df.loc[plot_df.Group.str.contains('Microbiome'), 'FPR']+.5e-2

    plot_df.loc[plot_df.Group.str.contains('Combined'), 'TPR'] = \
        plot_df.loc[plot_df.Group.str.contains('Combined'), 'TPR']-.5e-2

    plot_df.loc[plot_df.Group.str.contains('Combined'), 'FPR'] = \
        plot_df.loc[plot_df.Group.str.contains('Combined'), 'FPR']-.5e-2

    plt.figure(figsize=(8,8))
    ax=sns.lineplot(x='FPR', 
                 y='TPR', 
                 hue='Group', 
                 data=plot_df, 
                 linewidth=5, 
                 palette = {a:np.array( pal.as_hex() )[np.array([0,1, 2, 3, 6])][i]
                      for i,a in enumerate( plot_df.Group.unique() ) }
                 )
    plt.plot([0,1], 
             [0,1], 
             color='black', 
             linestyle='--',
             linewidth=5)
    plt.ylim([0,1.01])
    plt.xlim([-0.01,1])
    plt.xticks([-0.01,1])
    plt.yticks([0,1.01])
    ax.legend().set_title(None)
    ax.legend().remove() ## removing legend..

    plt.savefig(os.path.join(results_path, 
                             'Momspi-aurocs.pdf'), 
                format='pdf', 
                bbox_inches='tight', 
                dpi=900
                )

    plot_df = pd.concat([format_aupr_dfs(a, name_dict_pred_map, y_test) 
                         for a in name_dict_pred_map]
                        ).reset_index(drop=True)

    ### slight shifts to make al lines visible
    plot_df.loc[plot_df.Group.str.contains('Microbiome'), 'Precision'] = \
        plot_df.loc[plot_df.Group.str.contains('Microbiome'), 'Precision']+.5e-2

    plot_df.loc[plot_df.Group.str.contains('Microbiome'), 'Recall'] = \
        plot_df.loc[plot_df.Group.str.contains('Microbiome'), 'Recall']+.5e-2

    plot_df.loc[plot_df.Group.str.contains('Combined'), 'Precision'] = \
        plot_df.loc[plot_df.Group.str.contains('Combined'), 'Precision']-.5e-2

    plot_df.loc[plot_df.Group.str.contains('Combined'), 'FPR'] = \
        plot_df.loc[plot_df.Group.str.contains('Combined'), 'Precision']-.5e-2

    plt.figure(figsize=(8,8))
    ax=sns.lineplot(x='Recall', 
                    y='Precision', 
                    hue='Group', 
                    data=plot_df, 
                    linewidth=5, 
                    ci=0,
                    palette = {a:np.array( pal.as_hex() )[np.array([0,1, 2, 3, 6])][i]
                          for i,a in enumerate( plot_df.Group.unique() ) }
                 )
    cb=y_test.mean()
    sns.lineplot(x=[0, 1], 
                   y= [cb, cb], 
                   linewidth = 5, 
                   hue = ['Class balance ({:.2f})'.format(cb) ]*2,
                   palette={'Class balance ({:.2f})'.format(cb):'black'},
                   linestyle = '--', 
                   ax=ax)
    plt.ylim([0,1.01])
    plt.xlim([-0.01,1])
    plt.xticks([-0.01,1])
    plt.yticks([0,1.01])
    ax.legend().remove()
    plt.savefig(os.path.join(results_path, 
                             'Momspi-prs.pdf'), 
                format='pdf', 
                bbox_inches='tight', 
                dpi=900
                )

if __name__=='__main__':
    main()




        
        
    
    
    
    
    
    
    
    


