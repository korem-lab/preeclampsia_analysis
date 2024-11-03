import os 
CODE_DIR = os.path.dirname(os.path.realpath(__file__))


import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'axes.facecolor':'white', 
            'figure.facecolor':'white', 
            'axes.edgecolor':'black', 
            'grid.color': 'black'
            }, 
        style='ticks',
       font_scale=2)
import pylab as pl
from matplotlib.colors import SymLogNorm, Colormap
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from data_loading import load_data
from skbio.stats.composition import clr

import glob
cwarm = cm.get_cmap('coolwarm', 120)

cdict = {
    'red':   [(0.0, 0.0, 0.0),
              (0.2857, 1, 1),
              (0.5, 1.0, 1.0),
              (0.7143, 1.0, 1.0),
              (1.0, 0.5, 1.0)],

    'green': [(0.0, 0.0, 0.0),
              (0.2857, 1,1),
              (0.5, 1.0, 1.0),
              (0.7143, 1,1),
              (1.0, 0.0, 0.0)],

    'blue':  [(0.0, 1.0, 1.0),
              (0.2857, 1.0, 1.0),
              (0.5, 1.0, 1.0),
              (0.7143, 1, 1),
              (1.0, 0.0, 0.0)]
}

cwarm = LinearSegmentedColormap('custom_bwr', cdict)


qq_dict = {'micro':'purple', 
           'luminex':'orange', 
           'clinical':'green'}

color_hexmap = {2:'#F58A33',
                1:'#BFBEBE',
                0:'#3A8ECA'}

def main():
    data_path=os.path.join(CODE_DIR,'../../data/')

    md, micro_relabund, \
        luminex_processed, clin_processed = load_data(data_path=data_path)

    high_abund_cols=(micro_relabund>1e-3).mean(axis=0) > 0.05
    micro_clr = pd.DataFrame(clr(1e-8 + micro_relabund.loc[:, high_abund_cols].values ), 
                             index=micro_relabund.index, 
                             columns=micro_relabund.columns[high_abund_cols]
                            )
    
    
    dir_inds_map = {'bmi-less25':md.BMI<25 ,
                    'all-cohort':md.index, 
                    'bmi-greater25':md.BMI>=25}

    for path in glob.glob('../../results/multiomic-associations/*/diablo-graph.csv'):
        net=pd.read_csv(path, index_col=0)
        net['correl__tr'] = net['weight']
        
        G=nx.from_pandas_edgelist(net, 
                                  source='from_name', 
                                  target='to_name',
                                  edge_attr='correl__tr'
                                  )
        
        # extract the edge weight
        edge_colors = [a['correl__tr'] for u,v,a in G.edges(data=True)]
        
        
        ref_nodes = pd.Series( list(G.nodes) 
                         ).str.replace('_micro', '')\
                          .str.replace('_luminex', '')\
                          .str.replace('_clinical', '')

        all_colors = []

        inds=dir_inds_map[path.split('/')[-2]]
        for X in [micro_clr.loc[inds], 
                  clin_processed.loc[inds], 
                  luminex_processed.loc[inds]
                     ]:

            df_tmp = pd.concat([X.loc[:, 
                          X.columns.str.replace(':', '.')\
                                   .str.replace('-', '.').str.replace(' ', '.')\
                                   .str.replace('/', '.')
                          .isin( ref_nodes ) ], 
                       md.loc[inds].PEC_case], axis=1
                              )

            df_tmp = df_tmp.groupby('PEC_case')[df_tmp.columns[:-1]].median().T
            df_tmp['Color'] = 2*( df_tmp[True] > df_tmp[False] ) + \
                                    1*( df_tmp[True] == df_tmp[False]  ) 

            all_colors.append(df_tmp.copy())

        all_colors = pd.concat(all_colors)

        all_colors.index = all_colors.index\
                                    .str.replace(':', '.')\
                                    .str.replace('-', '.')\
                                    .str.replace(' ', '.')\
                                    .str.replace('/', '.')

        np.random.seed(1)
        plt.figure(figsize=(8,8))
        nx.draw(G, 
                with_labels=True, 
                node_size=1000,  
                linewidths=2, 
                edgecolors='black',
                edge_color=edge_colors,
                edge_cmap=cwarm, 
                edge_vmin=-1, 
                edge_vmax=1,
                node_color=[color_hexmap[a] for a in
                            all_colors.loc[ref_nodes].Color], 
                width=15,
                font_size=20
                )
        plt.savefig(path.replace('.csv', '-networkx.pdf'), 
                    dpi=900, 
                    format='pdf', 
                    )


        np.random.seed(1)
        plt.figure(figsize=(8,8))
        nx.draw(G, 
                with_labels=False, 
                node_size=1000,  
                linewidths=2, 
                edgecolors='black',
                edge_color=edge_colors,
                edge_cmap=cwarm, 
                edge_vmin=-1, 
                edge_vmax=1,
                node_color=[color_hexmap[a] for a in
                            all_colors.loc[ref_nodes].Color],
                width=15,
                font_size=20
                )
        plt.savefig(path.replace('.csv', '-networkx-no-text.pdf'), 
                    dpi=900, 
                    format='pdf', 
                    bbox_inches='tight'
                    )
        
        
        a = np.array([[0,1]])
        pl.figure(figsize=(9, 1.5))
        img = pl.imshow(a, 
                        cmap='coolwarm',
                        vmin=-1, 
                        vmax=1
                       )
        
        pl.gca().set_visible(False)
        cax = pl.axes([0.1, 0.2, 0.8, 0.6])
        pl.colorbar(orientation="horizontal", cax=cax)
        pl.xticks(fontsize=0)
        pl.savefig('../../results/multiomic-associations/colorbar.pdf', 
                   dpi=900, 
                   bbox_inches='tight',
                   format='pdf'
                   )
        
        
        
        ### make the loadings plots
        loadings = pd.read_csv( path.replace('graph', 'loadings'), index_col=0 )

        name_df_dict = {'micro':micro_relabund.columns.str.replace(':', '.')\
                                   .str.replace('-', '.').str.replace(' ', '.')\
                                   .str.replace('/', '.'), 
                        'luminex':luminex_processed.columns.str.replace(':', '.')\
                                   .str.replace('-', '.').str.replace(' ', '.')\
                                   .str.replace('/', '.'), 
                        'clinical':clin_processed.columns.str.replace(':', '.')\
                                   .str.replace('-', '.').str.replace(' ', '.')\
                                   .str.replace('/', '.')}

        name_full_df_dict={'micro':micro_clr, 
                           'luminex':luminex_processed, 
                           'clinical':clin_processed
                           }

        inds=dir_inds_map[path.split('/')[-2]]
        for grp in [ 'micro', 
                     'luminex', 
                     'clinical']:


                l_tmp = loadings.loc[loadings.index.isin(name_df_dict[grp])]\
                                .abs().sort_values(by='comp1')

                X = name_full_df_dict[grp].loc[inds]
                df_tmp = pd.concat([X.loc[:, 
                          X.columns.str.replace(':', '.')\
                                   .str.replace('-', '.').str.replace(' ', '.')\
                                   .str.replace('/', '.')
                          .isin( l_tmp.index ) ], 
                       md.loc[inds].PEC_case], axis=1
                              )

                df_tmp = df_tmp.groupby('PEC_case')[df_tmp.columns[:-1]].median().T
                df_tmp['Color'] = 2*( df_tmp[True] > df_tmp[False] ) + \
                                        1*( df_tmp[True] == df_tmp[False] )

                df_tmp.index=df_tmp.index.str.replace(':', '.')\
                                   .str.replace('-', '.').str.replace(' ', '.')\
                                   .str.replace('/', '.')


                plt.figure(figsize=(4,8))
                ax=sns.barplot(y=l_tmp.index, 
                               x=l_tmp.comp1.values, 
                               hue = df_tmp.loc[l_tmp.index].Color.values,
                               palette=color_hexmap, 
                               dodge=False
                              )
                plt.xticks([0, 0.5])
                ax.legend().remove()
                plt.savefig(path.replace('graph.csv', 
                                         'loading-boxplot-{}.pdf'.format(grp)
                                        ), 
                            dpi=900, 
                            bbox_inches='tight', 
                            format='pdf'
                           )

        
        
    a = np.array([[0,1]])
    pl.figure(figsize=(9, 1.5))
    img = pl.imshow(a, 
                    cmap=cwarm,
                    vmin=-1, 
                    vmax=1
                   )
    
    pl.gca().set_visible(False)
    cax = pl.axes([0.1, 0.2, 0.8, 0.6])
    pl.colorbar(orientation="horizontal", cax=cax)
    pl.xticks(fontsize=0)
    pl.savefig('../../results/multiomic-associations/colorbar.pdf', 
               dpi=900, 
               bbox_inches='tight',
               format='pdf'
               )


        
if __name__=='__main__':
    main()