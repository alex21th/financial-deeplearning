import torch
import pandas as pd
from time import time


class Dictionary():
    '''Builds a dictionary that maps the categorical variables defined in unique_values to the range
       [2, len(df.column.unique()) + 1] since indices 0 and 1 are reserved for <pad> and <unk>.'''
    def __init__(self, unique_values):
        self.idx2token = ['<pad>'] + ['<unk>'] + list(unique_values) # np.concatenate((np.array([-1, -1]), unique_values))
        self.token2idx = {}
        for idx, token in enumerate(self.idx2token):
            self.token2idx[token] = idx
    
    def add_token(self, token):
        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        return self.token2idx[token]

    def __len__(self):
        return len(self.idx2token)

    
class SegmentationDataset(torch.utils.data.Dataset):
    '''Mobility Segmentation dataset for PyTorch'''
    
    def __init__(self, df, predictor_variables, target_variables, max_length):
        '''Initialization'''
        self.df = df
        self.predictor_variables = predictor_variables
        self.target_variables = target_variables
        self.max_length = max_length
        
        self.hashes = self.df.hash.unique()
        
        self.hash2idx = dict()
        for idx, i in enumerate(self.hashes):
            self.hash2idx[i] = idx
            
        self.map_hash = [[] for i in self.hashes]
        for idx, i in enumerate(df.hash):
            self.map_hash[self.hash2idx[i]].append(idx)

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.hashes)

    def __getitem__(self, index):
        '''Generates one sample of data'''
#        print(index)
#        t_0 = time()
        itinerary = self.df.iloc[self.map_hash[index]]
#         current_hash = self.hashes[index]
#         itinerary = self.df[self.df.hash == current_hash]
        itinerary = itinerary.iloc[:self.max_length, :]
        length = itinerary.shape[0] - 1 # Account for the trailing step of the itinerary used for prediction
        
        X = itinerary.iloc[:-1, :][self.predictor_variables]
        X = torch.from_numpy(X.values).float()
        y = itinerary.iloc[1:, :][self.target_variables]
        y = torch.from_numpy(y.values).float()

        current_hash = self.hashes[index]
#        print(f'Getitem done {time() - t_0:.5f}')
        return X, length, y, current_hash
    

class PadSequence(object):
    '''Utility class to pad the variable length itineraries in SegmentationDataset.'''
    def __call__(self, batch):
        '''Callable method for the PadSequence class.
        
        Args: 
            batch (list): Receives a list of tuples generated from the __getitem__ method
                in the SegmentationDataset class. The list has length batch_size and each
                tuple is made of (X, length, y, current_hash)
           
        '''
        X = [elem[0] for elem in batch]
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=False)
        
        y = [elem[2] for elem in batch]
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=False)
        
        lengths = torch.tensor([elem[1] for elem in batch])
        hashes = [elem[3] for elem in batch]
        
        return X, lengths, y, hashes
