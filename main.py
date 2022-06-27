import torch.cuda

from skip_connection_model import *
from utils.decode import Decoder

print(torch.cuda.is_available())

decoder = Decoder(1296, gloss_dict='gloss_dict.pkl', search_mode='max')
model = CSLR(1024, 1296, 512, decoder)
# prefix = '../phoenix2014_data/features/trackedRightHand-92x132px'
prefix = '../phoenix2014_data/features/fullFrame-224x224px'
train_model(model, 'train', prefix, '../data.pkl', 'gloss_dict.pkl', 50000, 2, 5e-5, 1e-2, 'skip_model.pt')

