
import os 
CODE_DIR = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV,LassoCV,LogisticRegression
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.model_selection import LeaveOneOut,StratifiedKFold,KFold,RepeatedStratifiedKFold,RepeatedKFold,GridSearchCV
from skbio.stats.composition import clr
from sklearn.pipeline import Pipeline
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score

from rebalancedcv import RebalancedLeaveOneOut, RebalancedKFold
    
def rescale(micro):
     return( micro/(micro.sum(axis=1).values[:, np.newaxis] ) )

def mv_pseudocount(tbl):
    return np.power(10, np.floor(np.log10(rescale(tbl)[tbl>0].min().min())))

import warnings
warnings.filterwarnings('ignore')


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
    
    clin_reduced = pd.concat([clin_reduced,clin_full[['weight_in_kg',
                                                      'mean_arterial_pressure',
                                                      'systolic_bp',
                                                      'diastolic_bp']
                                                      ]],axis=1)
    
    
    
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

def get_tuned_predictor(X_tra,
                        y_tra, 
                        ):
    
    ### tuning/training pipeline used in all predictive analyses
    pipe = Pipeline([
                    ('skb', SelectKBest()),
                    ('lrg', LogisticRegression())
                    ])
    params = {
              'skb__k': [2,5,10],
              'lrg__C': np.logspace(0,8,10),
              }
    
    cv_model = GridSearchCV(pipe,
                            params,
                            scoring='roc_auc',
                            cv=5,
                            )

    cv_model.fit( X_tra, y_tra )
    
    return(cv_model)

from sklearn.utils import (
    _approximate_mode,
    _safe_indexing,
    check_random_state,
    indexable,
)
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import _num_samples, check_array, column_or_1d
import numpy as np

import numbers
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.multiclass import type_of_target
from abc import ABCMeta, abstractmethod
from itertools import chain, combinations

def flatten(xss):
    return [x for xs in xss for x in xs]

class RebalancedLeaveOneOutGroupMatching(BaseCrossValidator):
    """Rebalanced Leave-One-Out cross-validator, as described in `Austin et al.`

    Provides train/test indices to split data in train/test sets. Each
    sample is used once as a test set (singleton) while the remaining
    samples are used to form the training set, 
    with subsampling to ensure consistent class balances across all splits.

    This class is designed to have the same functionality and 
    implementation structure as scikit-learn's ``LeaveOneOut()``
    
    At least two observations per class are needed for `RebalancedLeaveOneOut`

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import LeaveOneOut
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 1, 1, 0])
    >>> rloo = RebalancedLeaveOneOut()
    >>> rloo.get_n_splits(X)
    2
    >>> print(rloo)
    RebalancedLeaveOneOut()
    >>> for i, (train_index, test_index) in enumerate(rloo.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[1, 2]
      Test:  index=[0]
    Fold 1:
      Train: index=[1, 4]
      Test:  index=[1]
    """
    
    def split(self, X, y, groups=None, seed=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
            
        seed : to enforce consistency in the subsampling

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        
        if seed is not None:
            np.random.seed(seed)
            
        X, y, groups = indexable(X, y, groups)
        
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            
            ## drop one sample with a `y` different from the test index
            susbample_inds = train_index != train_index[ 
                    np.random.choice(  np.where( (y[train_index] 
                                                         != y[test_index][0]
                                                 )&(
                                                groups[train_index]
                                                    ==groups[test_index][0]
                                                    )
                                               )[0] 
                                    
                                    
                                    ) ]
            train_index=train_index[susbample_inds]
            test_index = indices[test_index]
            yield train_index, test_index

    def _iter_test_indices(self, X, y, groups=None):
        n_samples = _num_samples(X)
        if n_samples <= 1:
            raise ValueError(
                "Cannot perform LeaveOneOut with n_samples={}.".format(n_samples)
            )
        return range(n_samples)

    def get_n_splits(self, X, y, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : object
            Needed to maintin class balance consistency.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return _num_samples(X)


def obtain_cv_predictions(X, y, md):
    preds = []
    trues = []
    outer_cv = RebalancedLeaveOneOutGroupMatching()
    split_or_not=[]
    for train_inds, test_inds in outer_cv.split(X, 
                                                y, 
                                                md.BMIg25.values
                                                ):
        
        X_tra, X_test =  X[train_inds], X[test_inds]
        y_tra, y_test = y[train_inds], y[test_inds]
        ## Test inner cv of single-level model, uses all data
        inner_cv_single_level = get_tuned_predictor(X_tra, y_tra)

#         Test inner cv of multilevel model, uses BMI cohort data
#         Multilevel model trains only on those samples that are in same BMI group (< or >= 25) as test samples
        same_bmi_idx = md.BMIg25.values[train_inds] == md.BMIg25.values[test_inds]
        inner_cv_multilevel = get_tuned_predictor(
                                                  X_tra[same_bmi_idx], 
                                                  y_tra[same_bmi_idx], 
                                                  )
        
        ## Use inner cv score to select multilevel vs single-level
        ## In other words, single vs multilevel tuning can be viewed as a hyperparameter in and of itself
        multi_better = inner_cv_multilevel.best_score_ > \
                              inner_cv_single_level.best_score_
        split_or_not.append(multi_better)
        
        preds += [ inner_cv_multilevel.predict_proba(X[test_inds])[0,1] 
                    if multi_better
                else inner_cv_single_level.predict_proba(X[test_inds])[0,1] ]

#         preds += [inner_cv_single_level.predict_proba(X[test_inds])[0,1] ]
        trues += [y_test]

    print( 'Percent of folds selecting multilevel: {:.2f}'.format( np.mean(split_or_not) ) )
    return(np.hstack(trues), 
           np.hstack(preds)
           )


def main(data_path=os.path.join(CODE_DIR,'../../data/'), 
         results_path= os.path.join(CODE_DIR,'../../results/predictions/'),
         seed=1
        ):
    
     ## the data path is in the git repo
    md, micro, luminex, clin_reduced, clin_full = \
                        load_data(data_path=data_path)
    micro_clr = pd.DataFrame(clr(mv_pseudocount(micro)+micro), 
                             index=micro.index, 
                             columns=micro.columns
                             )
    
    luminex.index=md.index
    clin_reduced.index=md.index
    
    y=md.PEC_case.values
    md['BMIg25'] = md['BMI']>=25
    
    micro_clr=micro_clr.loc[md.index]
    luminex=luminex.loc[md.index]
    clin_reduced=clin_reduced.loc[md.index]
    
    results_df = md[['PEC_case', 'BMIg25']]
    col_nms = ['Microbiome', 'Luminex', 'Clinical']
    
    clin_reduced=clin_reduced.drop([
                                    'PctFedPoverty', 
                                    'Education', 
                                    'V1AF14'
                                    ], 
                                    axis=1)
    
    print(clin_reduced.columns)
    
    ## Runs cross-validation on microbiome, luminex, and clinical features one-by-one 
    ## All hyperparam tuning occurs on inner folds
    ## Note- should take ~~10 minutes to run
    for i,tbl in enumerate([micro_clr, luminex, clin_reduced]):
        np.random.seed(seed)
        trues, preds = obtain_cv_predictions(tbl.values,
                                             md.PEC_case.values, 
                                             md
                                             )
        print(col_nms[i])
        print(roc_auc_score(trues, preds))
        results_df[col_nms[i]]=preds
    
    results_df['Combined'] = results_df.iloc[:, -3:].sum(axis=1)
        
    results_df.to_csv(
            os.path.join(os.path.join(results_path, 
                                      'all_predictions.csv')
                        ) )
    
    
if __name__=='__main__':
    main()





