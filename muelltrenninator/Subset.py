from torch.utils.data import Dataset

class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        """
        Parameters
        ----------

        dataset : Dataset
            The base dataset.

        indices : list

            List of the indices of the pictures the dataset should obtain.

        transform : transforms
            List of all the transforms that should be applied to the images.
        
        """
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):

        image, labels = self.dataset[self.indices[idx]]
        image = self.transform(image)

        return image, labels