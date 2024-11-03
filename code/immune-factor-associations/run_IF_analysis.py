
import os 
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from skbio.stats.distance import permanova, DistanceMatrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV,LassoCV,LogisticRegression, LinearRegression
from sklearn.feature_selection import SelectKBest,SelectPercentile
from scipy.stats import mannwhitneyu
from skbio.stats.composition import clr
from sklearn.pipeline import Pipeline
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.stats.multitest import fdrcorrection
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import zscore, ttest_ind
from statsmodels.stats.multitest import multipletests


sns.set(rc={'axes.facecolor':'white', 
            'figure.facecolor':'white', 
            'axes.edgecolor':'black', 
            'grid.color': 'black', 
            'axes.grid': False
            }, 
        
        style='ticks',
        font_scale=2
        )

def rescale(micro):
     return( micro/(micro.sum(axis=1).values[:, np.newaxis] ) )

def mv_pseudocount(tbl):
    return np.power(10, np.floor(np.log10(rescale(tbl)[tbl>0].min().min())))

import warnings
warnings.filterwarnings('ignore')

def add_15_percent_to_top_ylim(ax):
    ylim=ax.get_ylim() 
    ax.set_ylim( ylim[0], ylim[1] + ( ylim[1] - ylim[0] ) * .15)
    return(None)

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

    # md=md.loc[md.index.isin(micro.index)] ## WK
    md['PEC_32']=(md[md['PEC_any']==1]['GA_days_at_diagnosis']<(7*32)).astype(int)
    md.loc[md['PEC_any']==0,'PEC_32']=0

    md['PEC_34']=(md[md['PEC_any']==1]['GA_days_at_diagnosis']<(7*34)).astype(int)
    md.loc[md['PEC_any']==0,'PEC_34']=0

    # micro=micro.loc[md.index] ## WK
    
    print(micro.shape)
    
    ## prevalence based filtering
    micro=micro.loc[:, ( rescale(micro) > 0).mean(axis=0) > 0.1]
    print(micro.shape)
    micro_relabund = micro/micro.values.sum(axis=1)[:, np.newaxis]
    
    luminex=pd.read_csv(os.path.join(data_path, 
                                     'luminex_lod_by_batch_124.csv'),
                        index_col=0).rename(index={v:k for k,v in md.level_0.to_dict().items()})

    # luminex=luminex.loc[md.index] ## WK
    clin_reduced=pd.read_csv(os.path.join(data_path,
                                  'clin_reduced_124.csv'), 
                                  index_col=0).loc[md.level_0]
    clin_full=pd.read_csv(os.path.join(data_path,
                                  'clin_full_124.csv'), 
                                  index_col=0).loc[md.level_0]
    
    # increases performance of clinical from 0.61 to 0.71
    clin_reduced = pd.concat([clin_reduced,clin_full[['weight_in_kg','mean_arterial_pressure','systolic_bp','diastolic_bp']]],axis=1)


    
    
    ## scale the luminex data (by batch)
    ss=StandardScaler()
    luminex_processed = pd.concat([
        pd.DataFrame(ss.fit_transform(luminex.loc[md[md['batch_2']=='batch_1_2'].index]), index=md[md['batch_2']=='batch_1_2'].index,columns=luminex.columns).fillna(0),
        pd.DataFrame(ss.fit_transform(luminex.loc[md[md['batch_2']=='batch_3'].index]), index=md[md['batch_2']=='batch_3'].index,columns=luminex.columns).fillna(0),
    ]).loc[md.index]
    
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
           clin_full_processed
           )

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)



def run_luminex_DA(luminex_df, md, md_col, thresh=5, test='mwu'):
    res = []

    md_cases = md[md[md_col] == True]
    md_controls = md[md[md_col] == False]

    lum_cases = luminex_df.loc[md_cases.index]
    lum_controls = luminex_df.loc[md_controls.index]

    for c in luminex_df.columns:
        arr1 = lum_controls[c].dropna()
        arr2 = lum_cases[c].dropna()

        if (len(arr1) > thresh) and (len(arr2) > thresh):
            if test == 'mwu':
                stat, p = mannwhitneyu(arr1, arr2, alternative='two-sided')
            elif test == 'ttest':
                stat, p = ttest_ind(arr1, arr2)
                # stat, p = mannwhitneyu(arr1, arr2)
            stat = cohen_d(arr2, arr1) ## NOT SURE WHY BUT HAVE TO FLIP arr2 and arr1 here
        else:
            stat, p = np.nan, np.nan
            
        if np.median(arr2) > np.median(arr1):
            stat = stat*-1
        
        res.append((c, p, stat))

    res = pd.DataFrame(res, columns=['feature', 'p', 'stat']).set_index("feature")
    temp1 = res[~res['p'].isna()]
    temp1['q'] = multipletests(temp1['p'], 
                               method='fdr_bh', 
                              alpha=0.2
                              )[1]
    temp2 = res[res['p'].isna()]
    temp2['q'] = np.nan
    res = pd.concat([temp1, temp2])
    res.sort_values(by='p')
    return res





def confidence_ellipse(x, y, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, 
                      linewidth=5,
#                       palette={True:'gold', False:'lightblue'}
                      **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def make_IF_volcano(luminex, 
                    md):
    ## Volcano plot
    res_124_yes_impute = run_luminex_DA(luminex, 
                                        md, 
                                        'PEC_case', 
                                        thresh=5, 
                                        test='ttest'
                                        )

    res_124_yes_impute['signed_log(p)'] = np.log10(res_124_yes_impute['p']) \
                    * np.sign(res_124_yes_impute['stat'])
    res_124_yes_impute = res_124_yes_impute.sort_values(by='p')
    print(md.shape)
    print(res_124_yes_impute)

    x = res_124_yes_impute['q'].sort_values()
    y = res_124_yes_impute['p'].sort_values()
    
    xvals = [0.1, 0.2]
    yinterp = np.interp(xvals, x, y)

    FDR_vals = yinterp
    x = res_124_yes_impute['stat']

    mw_test = pd.DataFrame({'ps':res_124_yes_impute.p, 
                 'otu':luminex.columns, 
                 'fdr':fdrcorrection(res_124_yes_impute.p, alpha=.2)[1]
                 })\
        .sort_values('ps')
    mw_test['effect_size'] = x



    mw_test['negativelog10p']=-np.log10(mw_test['ps'])
    
    print(mw_test)
    print( (mw_test.effect_size>=0).mean() )
    print( (mw_test.effect_size>=0).sum() )
    print( (mw_test.effect_size<0).sum() )

    plt.figure(figsize=(8,8))
    ax=sns.scatterplot(x='effect_size', 
                       y='negativelog10p',
                    data=mw_test, 
                       s=100,
                       color='black'
                    )
    axxlim=ax.get_xlim()
    try:
        f_thresh=0.2
        qwe=np.array( [ mw_test.loc[mw_test['fdr']<f_thresh]['fdr'].max(), 
                        mw_test.loc[mw_test['fdr']>f_thresh]['fdr'].min() ] ).reshape(-1, 1)


        tts=LinearRegression().fit(qwe, 
                               [mw_test.loc[mw_test['fdr']<f_thresh]['negativelog10p'].min(), 
                                mw_test.loc[mw_test['fdr']>f_thresh]['negativelog10p'].max()]
                              ).predict(np.array([[f_thresh]]))[0]




        sns.lineplot(x=[ axxlim[0]-10, 
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
    except:
        pass
    
    try:
        f_thresh=0.1
        qwe=np.array( [ mw_test.loc[mw_test['fdr']<f_thresh]['fdr'].max(), 
                        mw_test.loc[mw_test['fdr']>f_thresh]['fdr'].min() ] ).reshape(-1, 1)


        tts=LinearRegression().fit(qwe, 
                               [mw_test.loc[mw_test['fdr']<f_thresh]['negativelog10p'].min(), 
                                mw_test.loc[mw_test['fdr']>f_thresh]['negativelog10p'].max()]
                              ).predict(np.array([[f_thresh]]))[0]


        sns.lineplot(x=[ axxlim[0]-10, 
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
    except:
        pass
    
    plt.xlim(axxlim)
    return(ax)



def main(data_path='../../data/',
         out_path='../../results/immune-factor-associations/',
         use_confidence_ellipse=False,
         seed=1):
    
    np.random.seed(seed)
    md, micro, luminex, clin_reduced, clin_full = load_data(data_path=data_path)
    print(luminex.shape)
    print(md['PEC_case'].value_counts())

    
    ### PCA scatterplot
    pca = PCA(n_components=2)
    X = pca.fit_transform(luminex)
    fig = plt.figure(figsize = (8,8))
    kwargs = {'x': X[:,0], 'y':X[:,1]}

    sns.scatterplot(**kwargs,
                    s=100, 
                    ax = plt.subplot(111), 
                    hue=md.PEC_case,
                    palette={True:'gold', False:'lightblue'}
                    )
    plt.xlabel("PC1, EV = {:.0f}%".format(pca.explained_variance_ratio_[0]*100))
    plt.ylabel("PC2, EV = {:.0f}%".format(pca.explained_variance_ratio_[1]*100))
    plt.legend(bbox_to_anchor=(1.45,1))


    print(pca.explained_variance_ratio_)

    dm = squareform(pdist(luminex, metric='euclidean'))
    print(permanova(DistanceMatrix(dm), md['PEC_case'], permutations=9999))
    
    
    temp = pd.DataFrame(X, index=md.index)
    temp['PEC_case'] = md['PEC_case']
    
    if use_confidence_ellipse:

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

    
    
    ## Fig 1B -- immune factor PC1 association with PEC
    X_ = pd.DataFrame(X)
    X_.index = luminex.index
    X_.columns = ['PC1', 'PC2']
    X_ = X_.join(md)

    plt.figure(figsize=(4,8))
    X_['PEC_case'] = X_['PEC_case']
    ax=sns.boxplot(data=X_, 
                   x='PEC_case',
                   y='PC1', 
                   fliersize=0, 
                   palette={True:'gold', False:'lightblue'}
                  )
    sns.swarmplot(data=X_,
                  x='PEC_case',
                  y='PC1', 
                  color='black',
                  dodge=True,
                  s=8, 
                  ax=ax
                  )
    
    arr1 = X_[X_['PEC_case'] == True]['PC1']
    arr2 = X_[X_['PEC_case'] == False]['PC1']
    print(mannwhitneyu(arr1, arr2, alternative='two-sided'))

    y1, y2 = plt.ylim()
    plt.ylim(y1, y2*1.2)

    plt.xlabel("")
    plt.xticks([0, 1], ['Control', 'sPEC'])
    plt.ylabel("Immune factors PC1, EV = {:.0f}%"\
                       .format(pca.explained_variance_ratio_[0]*100))

    print("here")
    print(mannwhitneyu(arr1, arr2))
    plt.title('Mannwhitneyu p: {:.3e}\n'.format(mannwhitneyu(arr1, arr2, alternative='two-sided').pvalue))


    plt.savefig(os.path.join(out_path, 'PCA_boxplot.pdf'),
                format='pdf',
                dpi=900, 
                bbox_inches='tight'
                )

    
    ## Fig 1C -- immune factor PC
    corrs=pd.DataFrame(pca.components_[0],
                        index=luminex.columns, 
                        columns=['PC1'])

    corrs = corrs.sort_values(by='PC1', ascending=False)


    plt.figure(figsize=(4,8))
    x = list(corrs['PC1'])
    y = list(np.arange(0, corrs.shape[0]))
    y.reverse()
    plt.barh(y, x)

    ytick_labels = list(corrs.index)
    ytick_labels = [x.split(' ')[0] for x in ytick_labels]
    ytick_labels.reverse()

    ytick_labels = ['IL-1Ra' if x == 'IL-1RA' else x for x in ytick_labels]

    ytick_labels = ['IL-1α' if x == 'IL-1a' else x for x in ytick_labels]
    ytick_labels = ['TNFα' if x == 'TNFa' else x for x in ytick_labels]
    ytick_labels = ['TGFα' if x == 'TGFa' else x for x in ytick_labels]
    ytick_labels = ['IFNα2' if x == 'IFNa2' else x for x in ytick_labels]

    ytick_labels = ['IFNγ' if x == 'IFNg' else x for x in ytick_labels]

    ytick_labels = ['IL-1β' if x == 'IL-1b' else x for x in ytick_labels]

    plt.yticks(np.arange(0, corrs.shape[0]), ytick_labels)
    plt.xlim([-0.1, 0.4])
    plt.savefig(os.path.join(out_path, 'PCA_components.pdf'),
                format='pdf',
                dpi=900, 
                bbox_inches='tight'
                )

    ## Volcano plot
    res_124_yes_impute = run_luminex_DA(luminex, 
                                        md, 
                                        'PEC_case', 
                                        thresh=5, 
                                        test='ttest'
                                        )

    res_124_yes_impute['signed_log(p)'] = np.log10(res_124_yes_impute['p']) \
                    * np.sign(res_124_yes_impute['stat'])
    res_124_yes_impute = res_124_yes_impute.sort_values(by='p')
    print(res_124_yes_impute)
    # assert 1 == 0

    x = res_124_yes_impute['q'].sort_values()
    y = res_124_yes_impute['p'].sort_values()
    
    xvals = [0.1, 0.2]
    yinterp = np.interp(xvals, x, y)

    FDR_vals = yinterp
    x = res_124_yes_impute['stat']

    mw_test = pd.DataFrame({'ps':res_124_yes_impute.p, 
                 'otu':luminex.columns, 
                 'fdr':fdrcorrection(res_124_yes_impute.p, alpha=.2)[1]
                 })\
        .sort_values('ps')
    mw_test['effect_size'] = x



    mw_test['negativelog10p']=-np.log10(mw_test['ps'])

    plt.figure(figsize=(8,8))
    ax=sns.scatterplot(x='effect_size', 
                       y='negativelog10p',
                    data=mw_test, 
                       s=100,
                       color='black'
                    )

    f_thresh=0.2
    qwe=np.array( [ mw_test.loc[mw_test['fdr']<f_thresh]['fdr'].max(), 
                    mw_test.loc[mw_test['fdr']>f_thresh]['fdr'].min() ] ).reshape(-1, 1)
    
    
    tts=LinearRegression().fit(qwe, 
                           [mw_test.loc[mw_test['fdr']<f_thresh]['negativelog10p'].min(), 
                            mw_test.loc[mw_test['fdr']>f_thresh]['negativelog10p'].max()]
                          ).predict(np.array([[f_thresh]]))[0]


    
    axxlim=ax.get_xlim()
    sns.lineplot(x=[ axxlim[0]-10, 
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
    
    
    
    f_thresh=0.1
    qwe=np.array( [ mw_test.loc[mw_test['fdr']<f_thresh]['fdr'].max(), 
                    mw_test.loc[mw_test['fdr']>f_thresh]['fdr'].min() ] ).reshape(-1, 1)
    
    
    tts=LinearRegression().fit(qwe, 
                           [mw_test.loc[mw_test['fdr']<f_thresh]['negativelog10p'].min(), 
                            mw_test.loc[mw_test['fdr']>f_thresh]['negativelog10p'].max()]
                          ).predict(np.array([[f_thresh]]))[0]

    
    sns.lineplot(x=[ axxlim[0]-10, 
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

    plt.savefig(os.path.join(out_path, 'IF-volcano.pdf'),
                format='pdf',
                dpi=900, 
                bbox_inches='tight'
                )

    inds=md.BMI>=25
    ax=make_IF_volcano(luminex.loc[inds], 
                    md.loc[inds])
    plt.savefig(os.path.join(out_path, 'IF-volcano-BMIg25.pdf'),
                format='pdf',
                dpi=900, 
                bbox_inches='tight'
                )
    
    ax=make_IF_volcano(luminex.loc[~inds], 
                    md.loc[~inds])
    plt.savefig(os.path.join(out_path, 'IF-volcano-BMIl25.pdf'),
                format='pdf',
                dpi=900, 
                bbox_inches='tight'
                )
    
    
    ## immune factor boxplots ## Change from 3 to 4 with batch specific std
    print('here2')
    print(mw_test)
    sig = mw_test[mw_test['fdr'] <= 0.2].index.to_list()
    
    for col in sig:
        plt.figure(figsize=(4,8))
        ax = sns.boxplot(y=luminex[col], 
                         x=md['PEC_case'],
                         palette={True:'gold', False:'lightblue'}
               )

        sns.swarmplot(y=luminex[col], 
                  x=md['PEC_case'],
                  s=8, color='black', 
                  ax=ax
                 )
        plt.title(col + '\n p = {:.2e}'.format(mw_test.loc[col].ps))
        
        add_15_percent_to_top_ylim(ax)

        plt.savefig(os.path.join(out_path, '{}-boxplot.pdf'\
                                        .format(col.replace('/', '-'))
                                ),
                    format='pdf',
                    dpi=900, 
                    bbox_inches='tight'
                    )
    

    ## write sum(all features) plot
    luminex = pd.read_csv(os.path.join(data_path, "luminex_lod_by_batch_124.csv"), index_col=0)
    md = pd.read_csv(os.path.join(data_path, "md_124.csv"), index_col=0)


    lum_with_impute = luminex.fillna(luminex.mean(axis=0)) ## fill NAs to feature means
    lsum_with_impute = pd.DataFrame(lum_with_impute.sum(axis=1), 
                                    columns=['sum']).join(md['PEC_case'])
    arr1 = lsum_with_impute[lsum_with_impute['PEC_case'] == True]['sum']
    arr2 = lsum_with_impute[lsum_with_impute['PEC_case'] == False]['sum']
    pval=mannwhitneyu(arr1, arr2, alternative='two-sided').pvalue
    plt.figure(figsize=(4,8))
    plt.semilogy()
    ax=sns.boxplot(data=lsum_with_impute,
                   x='PEC_case', 
                   y='sum', 
                   palette={True:'gold', 
                            False:'lightblue'}, 
                   fliersize=0)
    sns.swarmplot(data=lsum_with_impute,
                  x='PEC_case',
                  y='sum', 
                  color='black', 
                  s=5, 
                  ax=ax
                  )
    plt.title('Mann-whitney p: {:.3e}\n'.format(pval))
    ax.set_yticks(ticks=np.logspace(2, 5, 4))
    ax.minorticks_off()
    plt.ylim(100, 1e5)
    
    plt.savefig(os.path.join(out_path, 'IF-sum-all-concentrations.pdf'),
                    format='pdf',
                    dpi=900, 
                    bbox_inches='tight'
                    )
    
if __name__=='__main__':
    main()


















