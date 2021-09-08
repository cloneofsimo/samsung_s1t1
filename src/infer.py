import torch
from train import GNNRegressor
from train import ChemDataModule

import pandas as pd

model = GNNRegressor()
dm = ChemDataModule()
dm.setup('fit')

checkpoint_path = 'checkpoints/sample-gnn-epoch=162-val_mae_loss=0.14.ckpt'
state_dict = torch.load(checkpoint_path)['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k[6:]]=v

model.model.load_state_dict(new_state_dict)
model.model.eval()

ans = []
for x in dm.test_dataloader():
    y = model(x)
    ans += list(y.view(-1).detach().numpy())

submission = pd.read_csv('../../raw/sample_submission.csv')
submission['ST1_GAP(eV)'] = ans
submission.to_csv('submission.csv', index=False)