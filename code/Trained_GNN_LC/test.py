'''
Project: GNN_LC

                                Test of GNN_LC

Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import sys
GNN_path = './Trained_GNN_LC/'
sys.path.append(GNN_path)
sys.path.append(GNN_path +'/utilities')
import pandas as pd
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, r2_score
from GNN_LC import GNN_lignin


data_file = 'butina_L.csv'
target = 'GGG log10_x_solub iterative 343.15'
prop = 'lignin'

df              = pd.read_csv('data/'+data_file)    

# Build molecule from SMILE
mol_column     = 'Molecule_Solvent'
smiles_column = 'UniqueSmiles'

df[mol_column] = df[smiles_column].apply(Chem.MolFromSmiles)

df_results = GNN_lignin(df, mol_column)

ensemble_model = df_results['GNN_prediction_lignin']

exp_values   = df[target]
y_true     = exp_values.values
y_pred     = ensemble_model.values

mae_ensemble  = mean_absolute_error(y_true, y_pred)
r2_ensemble   = r2_score(y_true, y_pred)

print('MAE  :' + str(mae_ensemble))
print('R2   :' + str(r2_ensemble))
