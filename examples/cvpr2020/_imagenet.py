import argparse
import os

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from _augs import get_imagenet_transform
from pathlib import Path
import json

try:
    import boto3
    from io import BytesIO

    ImageS3_Implemented = True

    class ImageS3(Dataset):
        def __init__(self, root, transform=None, endpoint=None, bucket_name=None):
            with open(Path(__file__).parent / 'imagenet_classes.json', 'r') as fp:
                self._imagenet_classes = json.load(fp)

            if isinstance(root, list):
                self._s3_urls = root

            elif root.endswith('.txt'):
                with open(root, 'r') as f:
                    self._s3_urls = [l.strip() for l in f.readlines()]
            else:
                raise ValueError('Root %s not supported for retrieving images from an s3 bucket' % root)

            # If endpoint is none, determine the end point from the file names
            if endpoint is None:
                endpoint = '/'.join(self._s3_urls[0].split('/')[:3])
                self._remove_endpoint = True

            else:
                self._remove_endpoint = False

            if bucket_name is not None:
                self._bucket_name = bucket_name
            else:
                self._bucket_name = self._s3_urls[0].split('/')[3]

            # Access the bucket anonymously
            self._s3 = boto3.client('s3', aws_access_key_id='',
                                    aws_secret_access_key='',
                                    region_name='us-east-2',
                                    endpoint_url=endpoint)

            self._s3._request_signer.sign = (lambda *args, **kwargs: None)

            self._transform = transform



        def __getitem__(self, index):
            if self._remove_endpoint:
                fn = '/'.join(self._s3_urls[index].split('/')[4:])
            else:
                fn = self._s3_urls[index]

            im_bytes = self._s3.get_object(Bucket=self._bucket_name, Key=fn)['Body'].read()
            im_s3 = Image.open(BytesIO(im_bytes))

            # Copy and close the connection with the original image in the cloud bucket. Additionally, convert any grayscale image to RGB (replicate it to have three channels)
            im = im_s3.copy().convert('RGB')
            im_s3.close()

            if self._transform is not None:
                im = self._transform(im)

            target = self._imagenet_classes[fn.split("/")[-2]]

            return im, target

        def __len__(self):
            return len(self._s3_urls)

    def get_ImageNet(data_dir='.', batch_size=1, val_batch_size=1, workers=0,
                    mode='training',
                    normalize=True,
                    **kwargs):
        prep_trans = get_imagenet_transform(mode, normalize, patch_size=224, **kwargs)

        if (isinstance(data_dir, list)
        and (data_dir[0].endswith('txt')
        or data_dir[0].startswith('s3')
        or data_dir[0].startswith('http'))
        or data_dir.endswith('txt')):
            if isinstance(data_dir, list) and data_dir[0].endswith('txt'):
                data_dir = data_dir[0]
            image_dataset = ImageS3
        else:
            image_dataset = ImageFolder
            data_dir = os.path.join(data_dir, 'ILSVRC/Data/CLS-LOC/test')

        # If testing the model, return the validation set from MNIST
        if mode != 'training':
            ds = image_dataset(root=data_dir, transform=prep_trans)
            test_queue = DataLoader(ds, batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=workers,
                                    pin_memory=True)
            return test_queue

        imagenet_data = image_dataset(root=data_dir, transform=prep_trans)
        train_size = int(len(imagenet_data) * 0.96)
        val_size = len(imagenet_data) - train_size
        train_ds, valid_ds = random_split(imagenet_data, (train_size, val_size))

        train_queue = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=workers,
                                pin_memory=True)
        valid_queue = DataLoader(valid_ds, batch_size=val_batch_size,
                                shuffle=False,
                                num_workers=workers,
                                pin_memory=True)
        
        return train_queue, valid_queue

except ModuleNotFoundError:
    print('Loading ImageNet from S3 bucket requires boto3 to be installed; however, it was not found. ImageNet from S3 not supported in this session.')
    ImageS3_Implemented = False
    ImageS3 = None
