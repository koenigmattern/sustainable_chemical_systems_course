'''
Project: GNN_LC

                                GNN_LC

Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import sys
GNN_path = './Trained_GNN_LC/'
sys.path.append(GNN_path)
sys.path.append(GNN_path +'/utilities')

import numpy as np
import os
from utilities.GNN_architecture import GNN, GNN_with_fp
import torch
from utilities.mol2graph import mol2torchdata, get_dataloader
from rdkit.Chem.rdchem import Mol as rdkitMol

hyp = { 'n_ensembles': 5,
        'num_layer':3,
         'drop_ratio':0.1,
         'conv_dim':50,
         'n_ms':64,
         'mlp_layers':3,
         'mlp_dims':[50, 25, 1]}

def GNN_lignin(df, mol_column):
    '''
    Predicts log solubility of lignin for a collection of molecules

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the collection of rdkit molecule objects for which 
        the prediction of log-solubility will be performed. It must contain one
        molecule per row. An identifier (e.g., SELFIES or SMILES) per each 
        molecule is advisable for keeping track of the predictions. 
    mol_column : str
        Name of the column in df containing the rdkit molecule objects

    Returns
    -------
    df_log_sol : pd.DataFrame
        A copy of the df containing the predictions of each individual model 
        and the mean prediction of the ensemble of models.

    '''

    
    
    dumb_target = 'dumb_target'
    df[dumb_target] = np.repeat(np.nan, df.shape[0])
    
    # Mol 2 graph
    graph_column = 'graph'
    y_scaler = None
    df[graph_column] = mol2torchdata(df, mol_column, dumb_target, y_scaler)
    
    indices = df.index.tolist()
    predict_loader = get_dataloader(df, indices, dumb_target, 
                                          graph_column,  
                                          batch_size=df.shape[0], 
                                          shuffle=False, drop_last=False)
    
    # Hyperparameters
    num_layer   = hyp['num_layer']
    drop_ratio  = hyp['drop_ratio']
    conv_dim    = hyp['conv_dim']
    n_ms        = hyp['n_ms']
    mlp_layers  = hyp['mlp_layers']
    mlp_dims    = hyp['mlp_dims']

    # Ensemble of models
    n_ensembles = hyp['n_ensembles']
    path        = os.getcwd()
    
    for e in range(1, n_ensembles+1):
        ## for use in PSEvolve:
        path_model_info = path + '\\' + 'Trained_GNN_LC' + '\\' + 'lignin' + '\\Ensemble_' + str(e)

        ## for test.py:
        #path_model_info = path + '\\lignin' + '\\Ensemble_' + str(e)
        model = GNN(num_layer=num_layer, drop_ratio=drop_ratio, conv_dim=conv_dim,
                   neurons_message=n_ms, mlp_layers=mlp_layers, mlp_dims=mlp_dims)
        
        #model.load_state_dict(torch.load(path_model_info + '/Ensemble_' + str(e) + '.pth'))
        model.load_state_dict(torch.load(path_model_info + '\\Ensemble_' + str(e) + '.pth'))


        model.eval()
        with torch.no_grad():
            for batch_solvent in predict_loader:
                with torch.no_grad():
                    y_pred = model(batch_solvent).numpy().reshape(-1,)
                
        df['Ensemble_'+str(e)] = y_pred


    Y_pred_total      = df.loc[:, 'Ensemble_1':'Ensemble_'+str(n_ensembles)].to_numpy()
    y_pred_total_mean = np.mean(Y_pred_total, axis=1)
    df['GNN_prediction_lignin'] = y_pred_total_mean

    df_log_sol = df.copy(deep=True)
    
    return df_log_sol


def GNN_lignin_with_fp(df, mol_column):
    '''
    Predicts log solubility of lignin for a collection of molecules

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the collection of rdkit molecule objects for which
        the prediction of log-solubility will be performed. It must contain one
        molecule per row. An identifier (e.g., SELFIES or SMILES) per each
        molecule is advisable for keeping track of the predictions.
    mol_column : str
        Name of the column in df containing the rdkit molecule objects
    prop : str
        Name of the property to predict: 'lignin' or 'cellulose'


    Returns
    -------
    df_log_sol : pd.DataFrame
        A copy of the df containing the predictions of each individual model
        and the mean prediction of the ensemble of models.

    '''
    for i, mol in enumerate(df[mol_column]):
        if type(mol) is not rdkitMol:
            raise Exception(f'mol in row {i} is not valid')

    dumb_target = 'dumb_target'
    df[dumb_target] = np.repeat(np.nan, df.shape[0])

    # Mol 2 graph
    graph_column = 'graph'
    y_scaler = None
    df[graph_column] = mol2torchdata(df, mol_column, dumb_target, y_scaler)

    indices = df.index.tolist()
    predict_loader = get_dataloader(df, indices, dumb_target,
                                    graph_column,
                                    batch_size=df.shape[0],
                                    shuffle=False, drop_last=False)

    # Hyperparameters
    num_layer = hyp['num_layer']
    drop_ratio = hyp['drop_ratio']
    conv_dim = hyp['conv_dim']
    n_ms = hyp['n_ms']
    mlp_layers = hyp['mlp_layers']
    mlp_dims = hyp['mlp_dims']

    # Ensemble of models
    n_ensembles = hyp['n_ensembles']
    path = os.getcwd()

    for e in range(1, n_ensembles + 1):
        # path_model_info = path + '/' + prop +'/Ensemble_' + str(e)
        path_model_info = path + '\\' + 'Trained_GNN_LC' + '\\' + 'lignin' + '\\Ensemble_' + str(e)

        model = GNN_with_fp(num_layer=num_layer, drop_ratio=drop_ratio, conv_dim=conv_dim,
                    neurons_message=n_ms, mlp_layers=mlp_layers, mlp_dims=mlp_dims)

        # model.load_state_dict(torch.load(path_model_info + '/Ensemble_' + str(e) + '.pth'))
        model.load_state_dict(torch.load(path_model_info + '\\Ensemble_' + str(e) + '.pth'))

        model.eval()
        with torch.no_grad():
            for batch_solvent in predict_loader:
                with torch.no_grad():
                    res_tup = model(batch_solvent)
                    y_pred = res_tup[0].numpy().reshape(-1, )
                    fp = res_tup[1].numpy().tolist()

        df['Ensemble_pred_' + str(e)] = y_pred
        df['Ensemble_fp_' + str(e)] = fp

    pred_col_names = df.filter(regex='Ensemble_pred_').columns.to_list()
    df['GNN_prediction_lignin'] = df[pred_col_names].mean(axis=1)


    fp_col_names = df.filter(regex='Ensemble_fp_').columns.to_list()
    fps = np.array(df[fp_col_names].values.tolist())
    mean_fps = fps.mean(axis=1)
    df['GNN_fp'] = mean_fps.tolist()
    df_log_sol = df.copy(deep=True)

    return df_log_sol

