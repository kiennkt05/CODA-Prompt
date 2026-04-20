from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import pickle
import torch
import torch.utils.data as data
from .utils import download_url, check_integrity
import random
import torchvision.datasets as datasets
import yaml

from pathlib import Path

try:
    import decord
    # Tell decord to return PyTorch tensors directly
    decord.bridge.set_bridge('torch')
except ImportError:
    raise ImportError("Please install decord: pip install decord")

def collate_video(batch):
    """
    Returns (B, S, C, H, W) videos and (B,) labels — no flattening.
    Segment averaging happens inside evaluate() after the forward pass.
    """
    videos, labels, task_ids = zip(*batch)
    return torch.stack(videos, dim=0), torch.as_tensor(labels, dtype=torch.long), torch.as_tensor(task_ids, dtype=torch.long)

class iDataset(data.Dataset):
    
    def __init__(self, root,
                train=True, transform=None,
                download_flag=False, lab=True, swap_dset = None, 
                tasks=None, seed=-1, rand_split=False, validation=False, kfolds=5):

        # process rest of args
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.validation = validation
        self.seed = seed
        self.t = -1
        self.tasks = tasks
        self.download_flag = download_flag

        # load dataset
        self.load()
        self.num_classes = len(np.unique(self.targets))

        # remap labels to match task order
        c = 0
        self.class_mapping = {}
        self.class_mapping[-1] = -1
        for task in self.tasks:
            for k in task:
                self.class_mapping[k] = c
                c += 1

        # targets as numpy.array
        self.data = np.asarray(self.data)
        self.targets = np.asarray(self.targets)

        # if validation
        if self.validation:
            
            # shuffle
            state = np.random.get_state()
            np.random.seed(self.seed)
            randomize = np.random.permutation(len(self.targets))
            self.data = self.data[randomize]
            self.targets = self.targets[randomize]
            np.random.set_state(state)

            # sample
            n_data = len(self.targets)
            if self.train:
                self.data = self.data[:int(0.8*n_data)]
                self.targets = self.targets[:int(0.8*n_data)]
            else:
                self.data = self.data[int(0.8*n_data):]
                self.targets = self.targets[int(0.8*n_data):]

            # train set
            if self.train:
                self.data = self.data[:int(0.8*n_data)]
                self.targets = self.targets[:int(0.8*n_data)]
                self.archive = []
                domain_i = 0
                for task in self.tasks:
                    if True:
                        locs = np.isin(self.targets, task).nonzero()[0]
                        self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))

            # val set
            else:
                self.archive = []
                domain_i = 0
                for task in self.tasks:
                    if True:
                        locs = np.isin(self.targets, task).nonzero()[0]
                        self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))

        # else
        else:
            self.archive = []
            domain_i = 0
            for task in self.tasks:
                if True:
                    locs = np.isin(self.targets, task).nonzero()[0]
                    self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))

        if self.train:
            self.coreset = (np.zeros(0, dtype=self.data.dtype), np.zeros(0, dtype=self.targets.dtype))

    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.class_mapping[target], self.t

    def load_dataset(self, t, train=True):
        
        if train:
            self.data, self.targets = self.archive[t] 
        else:
            self.data    = np.concatenate([self.archive[s][0] for s in range(t+1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t+1)], axis=0)
        self.t = t

    def append_coreset(self, only=False, interp=False):
        len_core = len(self.coreset[0])
        if self.train and (len_core > 0):
            if only:
                self.data, self.targets = self.coreset
            else:
                len_data = len(self.data)
                sample_ind = np.random.choice(len_core, len_data)
                self.data = np.concatenate([self.data, self.coreset[0][sample_ind]], axis=0)
                self.targets = np.concatenate([self.targets, self.coreset[1][sample_ind]], axis=0)

    def update_coreset(self, coreset_size, seen):
        num_data_per = coreset_size // len(seen)
        remainder = coreset_size % len(seen)
        data = []
        targets = []
        
        # random coreset management; latest classes take memory remainder
        # coreset selection without affecting RNG state
        state = np.random.get_state()
        np.random.seed(self.seed)
        for k in reversed(seen):
            mapped_targets = [self.class_mapping[self.targets[i]] for i in range(len(self.targets))]
            locs = (mapped_targets == k).nonzero()[0]
            if (remainder > 0) and (len(locs) > num_data_per):
                num_data_k = num_data_per + 1
                remainder -= 1
            else:
                num_data_k = min(len(locs), num_data_per)
            locs_chosen = locs[np.random.choice(len(locs), num_data_k, replace=False)]
            data.append([self.data[loc] for loc in locs_chosen])
            targets.append([self.targets[loc] for loc in locs_chosen])
        self.coreset = (np.concatenate(list(reversed(data)), axis=0), np.concatenate(list(reversed(targets)), axis=0))
        np.random.set_state(state)

    def load(self):
        pass

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class iCIFAR10(iDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iDataset Dataset.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    im_size=32
    nch=3

    def load(self):

        # download dataset
        if self.download_flag:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train or self.validation:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.course_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                if 'coarse_labels' in entry:
                    self.course_targets.extend(entry['coarse_labels'])
                
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

class iCIFAR100(iCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iCIFAR10 Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    im_size=32
    nch=3

class iIMAGENET_R(iDataset):
    
    base_folder = 'imagenet-r'
    im_size=224
    nch=3
    def load(self):

        # load splits from config file
        if self.train or self.validation:
            data_config = yaml.load(open('dataloaders/splits/imagenet-r_train.yaml', 'r'), Loader=yaml.Loader)
        else:
            data_config = yaml.load(open('dataloaders/splits/imagenet-r_test.yaml', 'r'), Loader=yaml.Loader)
        self.data = data_config['data']
        self.targets = data_config['targets']

    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img_path, target = self.data[index], self.targets[index]
        img = jpg_image_to_array(img_path)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.class_mapping[target], self.t

    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == 'train':
                parse_train_archive(self.root)
            elif self.split == 'val':
                parse_val_archive(self.root)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

class iDOMAIN_NET(iIMAGENET_R):
    base_folder = 'DomainNet'
    im_size=224
    nch=3
    def load(self):
        
        # load splits from config file
        if self.train or self.validation:
            data_config = yaml.load(open('dataloaders/splits/domainnet_train.yaml', 'r'), Loader=yaml.Loader)
        else:
            data_config = yaml.load(open('dataloaders/splits/domainnet_test.yaml', 'r'), Loader=yaml.Loader)
        self.data = data_config['data']
        self.targets = data_config['targets']

class iUCF101(iDataset):
    """
    Online Dataset for UCF101 Class-Incremental Learning for CODA-Prompt.
    """
    im_size = 224
    nch = 3

    def load(self):
        self.num_segments = 3
        self.video_root = Path(self.root) / "videos"

        if not self.video_root.exists():
            raise FileNotFoundError(f"Video directory not found at {self.video_root}")

        self._available_videos = {}
        for ext in ['.mp4', '.avi', '.mkv', '.webm']:
            for p in self.video_root.glob(f"*/*{ext}"):
                self._available_videos[f"{p.parent.name}/{p.stem}"] = p

        # Dynamically load the correct PKL based on max_tasks/num_tasks passed from trainer
        num_tasks = len(self.tasks) 
        pkl_file = Path(self.root) / f"UCF101_data_{num_tasks}tasks.pkl"
        with open(pkl_file, 'rb') as f:
            pkl_data = pickle.load(f)
            
        split_key = 'train' if self.train else 'test'
        
        self.classes = []
        seen = set()
        for task_dict in pkl_data["train"]:
            for cls_name in task_dict.keys():
                if cls_name not in seen:
                    self.classes.append(cls_name)
                    seen.add(cls_name)
                    
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.class_mask = [
            [self.class_to_idx[cls_name] for cls_name in task_dict.keys() if cls_name in self.class_to_idx]
            for task_dict in pkl_data["train"]
        ]
        
        # Override the tasks list passed from trainer.py to strictly match the PKL file
        self.tasks.clear()
        for mask in self.class_mask:
            self.tasks.append(mask)

        self.data = []
        self.targets = []
        self.task_ids = []
        
        for task_id, task_dict in enumerate(pkl_data[split_key]):
            for cls_name, entries in task_dict.items():
                if cls_name not in self.class_to_idx:
                    continue  
                
                label = self.class_to_idx[cls_name]
                
                for entry in entries:
                    video_path = self._available_videos.get(f"{cls_name}/{entry}")
                    if video_path is None:
                        continue
                        
                    self.data.append({
                        'video_path': str(video_path),
                        'cls_name': cls_name
                    })
                    self.targets.append(label)
                    self.task_ids.append(task_id)

    def __init__(self, root, train=True, transform=None, download_flag=False, lab=True, swap_dset=None, tasks=None, seed=-1, rand_split=False, validation=False, kfolds=5):
        # Let iDataset handle basic property setup and mapping
        super().__init__(root, train, transform, download_flag, lab, swap_dset, tasks, seed, rand_split, validation, kfolds)
        
        # EXPLICIT OVERRIDE: Re-build self.archive strictly using the task_ids array matching the PKL
        # This matches your exact requested Subset logic: [i for i, t_id in enumerate(dataset.task_ids) if t_id == task_id]
        self.archive = []
        for task_id in range(len(self.class_mask)):
            split_indices = [i for i, t_id in enumerate(self.task_ids) if t_id == task_id]
            self.archive.append((self.data[split_indices].copy(), self.targets[split_indices].copy()))

    def __getitem__(self, index, simple=False):
        sample = self.data[index]
        target = self.targets[index]
        
        vr = decord.VideoReader(sample['video_path'], ctx=decord.cpu(), num_threads=1)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / fps
        
        t_start = 0.0
        t_end = duration
        seg_duration = max(t_end - t_start, 0.1) / self.num_segments
        
        frame_indices = []
        for i in range(self.num_segments):
            s = t_start + i * seg_duration
            e = t_start + (i + 1) * seg_duration
            if self.train:
                t = random.uniform(s, e)
            else:
                t = (s + e) / 2.0
                
            f_idx = int(t * fps)
            f_idx = min(max(f_idx, 0), total_frames - 1)
            frame_indices.append(f_idx)
            
        frames_tensor = vr.get_batch(frame_indices)
        frames_pil = [Image.fromarray(frame.numpy()) for frame in frames_tensor]
        
        if self.transform is not None:
            frames_transformed = [self.transform(img) for img in frames_pil]
            video_tensor = torch.stack(frames_transformed)
        else:
            video_tensor = frames_tensor.permute(0, 3, 1, 2).float() / 255.0

        return video_tensor, self.class_mapping[target], self.t

class iActivityNet(iDataset):
    """
    Online Dataset for ActivityNet Class-Incremental Learning for CODA-Prompt.
    """
    im_size = 224
    nch = 3

    def load(self):
        self.num_segments = 3
        self.video_root = Path(self.root) / "AnetVideos"

        if not self.video_root.exists():
            raise FileNotFoundError(f"Video directory not found at {self.video_root}")

        self._available_videos = {}
        for ext in ['.mp4', '.avi', '.mkv', '.webm']:
            for p in self.video_root.glob(f"*{ext}"):
                self._available_videos[p.stem] = p

        num_tasks = len(self.tasks) 
        pkl_file = Path(self.root) / f"ActivityNet_data_{num_tasks}tasks.pkl"
        with open(pkl_file, 'rb') as f:
            pkl_data = pickle.load(f)
            
        split_key = 'train' if self.train else 'val'
        
        self.classes = []
        seen = set()
        for task_dict in pkl_data["train"]:
            for cls_name in task_dict.keys():
                if cls_name not in seen:
                    self.classes.append(cls_name)
                    seen.add(cls_name)
                    
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.class_mask = [
            [self.class_to_idx[cls_name] for cls_name in task_dict.keys() if cls_name in self.class_to_idx]
            for task_dict in pkl_data["train"]
        ]
        
        # Override the tasks list passed from trainer.py to strictly match the PKL file
        self.tasks.clear()
        for mask in self.class_mask:
            self.tasks.append(mask)

        self.data = []
        self.targets = []
        self.task_ids = []
        
        for task_id, task_dict in enumerate(pkl_data[split_key]):
            for cls_name, entries in task_dict.items():
                if cls_name not in self.class_to_idx:
                    continue  
                
                label = self.class_to_idx[cls_name]
                
                for entry in entries:
                    video_path = self._available_videos.get(entry['filename'])
                    if video_path is None:
                        continue
                        
                    self.data.append({
                        'video_path': str(video_path),
                        't_start': float(entry['t_start']),
                        't_end': float(entry['t_end']),
                        'cls_name': cls_name
                    })
                    self.targets.append(label)
                    self.task_ids.append(task_id)

    def __init__(self, root, train=True, transform=None, download_flag=False, lab=True, swap_dset=None, tasks=None, seed=-1, rand_split=False, validation=False, kfolds=5):
        # Let iDataset handle basic property setup and mapping
        super().__init__(root, train, transform, download_flag, lab, swap_dset, tasks, seed, rand_split, validation, kfolds)
        
        # EXPLICIT OVERRIDE: Re-build self.archive strictly using the task_ids array matching the PKL
        self.archive = []
        for task_id in range(len(self.class_mask)):
            split_indices = [i for i, t_id in enumerate(self.task_ids) if t_id == task_id]
            self.archive.append((self.data[split_indices].copy(), self.targets[split_indices].copy()))

    def __getitem__(self, index, simple=False):
        sample = self.data[index]
        target = self.targets[index]
        
        vr = decord.VideoReader(sample['video_path'], ctx=decord.cpu(), num_threads=1)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / fps
        
        t_start = max(0.0, sample['t_start'])
        t_end = min(duration, sample['t_end'])
        seg_duration = max(t_end - t_start, 0.1) / self.num_segments
        
        frame_indices = []
        for i in range(self.num_segments):
            s = t_start + i * seg_duration
            e = t_start + (i + 1) * seg_duration
            if self.train:
                t = random.uniform(s, e)
            else:
                t = (s + e) / 2.0
                
            f_idx = int(t * fps)
            f_idx = min(max(f_idx, 0), total_frames - 1)
            frame_indices.append(f_idx)
            
        frames_tensor = vr.get_batch(frame_indices)
        frames_pil = [Image.fromarray(frame.numpy()) for frame in frames_tensor]
        
        if self.transform is not None:
            frames_transformed = [self.transform(img) for img in frames_pil]
            video_tensor = torch.stack(frames_transformed)
        else:
            video_tensor = frames_tensor.permute(0, 3, 1, 2).float() / 255.0

        return video_tensor, self.class_mapping[target], self.t

def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape 
    (width, height, channels)
    """
    with Image.open(image_path) as image:      
        image = image.convert('RGB')
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
    return im_arr
