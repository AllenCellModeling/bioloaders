'A lot of this is copied from https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py'

from bioloaders import Dataset
import os
import os.path
import errno
import glob
import numpy as np
import bioloaders.utils as utils

from tqdm import tqdm

from scipy import misc

import pdb

class ic2d(Dataset):
    
    """`Integrated Cell 2D <downloads.allencell.org/publication-data/building-a-2d-integrated-cell/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in a Y,X,C image
            and returns a transformed version.
        label_type (string, optional): {'onehot', 'index', 'string'} specifies the output 
            type of the label.
    """
    
    
    urls = [
        'http://downloads.allencell.org/publication-data/building-a-2d-integrated-cell/ic2D.tar',
    ]
    
    info = ['downloads.allencell.org/publication-data/building-a-2d-integrated-cell/',
            'https://github.com/AllenCellModeling/torch_integrated_cell',
            """@article {johnson2017generative, \
            author={Gregory R. Johnson, Rory M. Donovan-Maiye, Mary M. Maleckar}, \
            title = {Generative Modeling with Conditional Autoencoders: Building an Integrated Cell}, \
            year = {2017}, \
            url={https://arxiv.org/abs/1705.00092}, \
            journal={arXiv preprint arXiv:1705.00092}, \ 
            }"""]
    
    def __init__(self, root, train=True, transform=None, label_type = 'onehot', buffer_images=True, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.label_type = label_type
        self.buffer_images = buffer_images
        
        if download:
            self.download()
            
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        
        
        image_paths = glob.glob(self.root + '/*/*.png')
        
        labels = [path.split('/')[-2] for path in image_paths]
        ulabels, labels_index = np.unique(labels, return_inverse=True)
        labels_onehot = utils.index2onehot(labels_index, len(ulabels))
        
        self.ulabels = ulabels
        self.labels = labels
        self.labels_index = labels_index
        self.labels_onehot = labels_onehot
        
        self.image_paths = image_paths
        
        if buffer_images:
            print('Buffering images')
            self.images = [misc.imread(path) for path in tqdm(image_paths)]
        
        
        
        
    def __getitem__(self, index):
        if self.buffer_images:
            image = self.images[index]
        else:
            image = misc.imread(self.image_paths[index])
            
        if self.label_type == 'string':
            label = self.labels[index]
        elif self.label_type == 'index':
            label = self.labels_index[index]
        elif self.label_type == 'onehot':
            label = self.labels_onehot[index]
            
        return {'image': image, 'label': label}
    
    def __len__(self):
        return len(self.image_paths)
    
    def _check_exists(self):
        #this could be done better
               
        return os.path.exists(self.root + os.sep + 'citation.bib')

    
    def download(self):
        """Download the data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import tarfile
    
        if self._check_exists():
                return
            
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
                
        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
                
            with tarfile.open(file_path) as zip_f:
                zip_f.extractall(path=self.root)
            os.unlink(file_path)
            
        print('Done!')
    
        pass
    
class ic3d(Dataset):

    urls = [
        'http://downloads.allencell.org/publication-data/building-a-3d-integrated-cell/ipp_17_10_25.tar',
    ]

    info = ['downloads.allencell.org/publication-data/building-a-3d-integrated-cell/',
            'https://github.com/AllenCellModeling/pytorch_integrated_cell',
            """@article {Johnson238378, \
            author = {Johnson, Gregory R. and Donovan-Maiye, Rory M. and Maleckar, Mary M.}, \
            title = {Building a 3D Integrated Cell}, \
            year = {2017}, \
            doi = {10.1101/238378}, \
            publisher = {Cold Spring Harbor Laboratory}, \
            URL = {https://www.biorxiv.org/content/early/2017/12/21/238378}, \
            eprint = {https://www.biorxiv.org/content/early/2017/12/21/238378.full.pdf}, \
            journal = {bioRxiv} \
            }"""]
    
    def __init__(self, root, train=True, download=False):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
    
    def download(self):
        pass