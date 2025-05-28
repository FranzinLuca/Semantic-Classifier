
import torch
from torch.utils.data import Dataset, DataLoader

def collate_and_offset_data(batch):
    """pad the batch to the max length
    """
    batch_pairs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    prop_ids = []
    claim_ids = []
    offsets = [0]

    for pairs in batch_pairs:
        for prop, claim in pairs:
            prop_ids.append(prop)
            claim_ids.append(claim)
        offsets.append(len(prop_ids))

    offsets.pop()
    
    statement_tensor = torch.tensor(prop_ids, dtype=torch.long)
    claim_tensor = torch.tensor(claim_ids, dtype=torch.long)
    offsets_tensor = torch.tensor(offsets, dtype=torch.long)
    labels_tensor = torch.tensor(targets, dtype=torch.long)

    return statement_tensor, claim_tensor, offsets_tensor, labels_tensor

# Define Claims dataset
class ClaimsDataset(Dataset):
    def __init__(self, dataset):
        self.data = []
        
        for _, row in dataset.iterrows():
            """add the properties to the dataset
            """
            pairs_data = row['pairs']
            label = row['label_int']
            
            if isinstance(pairs_data, list):
                pairs = pairs_data
            else:
                pairs = []
            
            self.data.append((pairs, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pairs, target = self.data[idx]
        return pairs, target

def create_dataloader(dataframe, BATCH_SIZE = 128, NUM_WORKERS = 1, shuffle=True):
    dataset = ClaimsDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collate_and_offset_data, num_workers=NUM_WORKERS)
    return dataloader 