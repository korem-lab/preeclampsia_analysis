
import os 
CODE_DIR = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

def format_auroc_dfs(prds, y_test):
    fpr, tpr, __ = roc_curve(y_test, 
                             prds, 
                             drop_intermediate=False
                             )
    area = auc(fpr, tpr)
    fpr += tpr*1e-7
    return( pd.DataFrame({'FPR':fpr, 
                          'TPR':tpr, 
                          'Group':prds.name + ' (auROC = {:.2f})'.format(area)}
                        )
          )

def format_aupr_dfs(prds, y_test):
    pre, rec, __ = precision_recall_curve(y_test, 
                                          prds
                                          )
    area = auc(rec, pre)
    rec -= pre*1e-7
    return( pd.DataFrame({'Precision':pre, 
                          'Recall':rec, 
                          'Group':prds.name + ' (auPR = {:.2f})'.format(area)}
                        )
          )


sns.set(rc={'axes.facecolor':'white', 
            'figure.facecolor':'white', 
            'axes.edgecolor':'black', 
            'grid.color': 'black'
            }, 
       font_scale=2)
pal = sns.color_palette()


def main(results_path= os.path.join(CODE_DIR,'../../results/predictions/')):
    res = pd.read_csv(os.path.join(results_path,
                                   'all_predictions.csv'),
                      index_col=0)

    res['Micro + Lum'] = res.Microbiome + res.Luminex
    res['All combined'] = res['Combined']
    res = res[['PEC_case', 'BMIg25', 'Microbiome', 
               'Luminex', 'Clinical', 'Micro + Lum', 'All combined']]


    plot_df = pd.concat([format_auroc_dfs(res[a],
                                          res['PEC_case']) 
                         for a in res.columns[2:]]
                        ).reset_index(drop=True)

    plt.figure(figsize=(8,8))
    ax=sns.lineplot(x='FPR', 
                    y='TPR', 
                    hue='Group', 
                    data=plot_df, 
                    linewidth=5, 
                    palette = {a:np.array( pal.as_hex() )[np.array([0,1,
                                                                    2,3,
                                                                    6])][i]
                      for i,a in enumerate( plot_df.Group.unique() ) }
                 )
    plt.plot([0,1], 
             [0,1], 
             color='black', 
             linestyle = '--', 
             linewidth=5)
    plt.ylim([0,1.01])
    plt.xlim([-0.01,1])
    plt.xticks([-0.01,1])
    plt.yticks([0,1.01])
    ax.legend().set_title(None)
    plt.legend().remove()
    plt.savefig(os.path.join(results_path, 
                             'overall_numom_cohort_roc.pdf'
                             ), 
                dpi=900, 
                bbox_inches='tight',
                format='pdf'
                )

    plot_df = pd.concat([format_aupr_dfs(res[a],
                                          res['PEC_case']) 
                         for a in res.columns[2:]]
                        ).reset_index(drop=True)

    plt.figure(figsize=(8,8))
    ax=sns.lineplot(x='Recall', 
                    y='Precision', 
                    hue='Group', 
                    data=plot_df, 
                    linewidth=5, 
                    palette = {a:np.array( pal.as_hex() )[np.array([0,1,
                                                                    2,3,
                                                                    6])][i]
                      for i,a in enumerate( plot_df.Group.unique() ) },
                    ci=0
                 )

    cb=res['PEC_case'].mean()
    sns.lineplot(x= [0, 1], y= [cb, cb], 
                   linewidth = 5, 
                   hue = ['Class balance ({:.2f})'.format(cb) ]*2,
                   palette={'Class balance ({:.2f})'.format(cb):'black'},
                   linestyle = '--', 
                   ax=ax)
    plt.ylim([0,1.01])
    plt.xlim([-0.01,1])
    plt.xticks([-0.01,1])
    plt.yticks([0,1.01])
    ax.legend().set_title(None)
    plt.legend().remove()
    plt.savefig(os.path.join(results_path, 
                             'overall_numom_cohort_pr.pdf'
                             ), 
                dpi=900, 
                bbox_inches='tight',
                format='pdf'
                )
    
if __name__=='__main__':
    main()

