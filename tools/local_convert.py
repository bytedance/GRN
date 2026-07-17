import torch


# GRN_ind_H_ep599.pth: /tmp/model_573902c96b167c5b8e58c658285f628b_slim.ckpt
cur_path = '/tmp/model_573902c96b167c5b8e58c658285f628b.ckpt'
save_path = cur_path.replace('.ckpt', '_slim.ckpt')
weights = torch.load(cur_path)
del weights['optimizer']
torch.save(weights, save_path)
print('done')
print(f'save to {save_path}')
        