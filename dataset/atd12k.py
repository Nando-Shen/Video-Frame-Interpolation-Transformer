import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import numpy as np

class ATD12k(Dataset):
    def __init__(self, data_root, is_training , input_frames="1357", mode='mini'):
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        """
        self.data_root = data_root
        self.region_root = data_root
        self.training = is_training
        self.inputs = input_frames
        self.result_dir = '/home/kuhu6123/jshe2377/12_ab/Video-Frame-Interpolation-Transformer/abregion12'
        if is_training:
            self.region_root = os.path.join(self.data_root, 'train_10k_region')
            self.data_root = os.path.join(self.data_root, 'train_10k')
        else:
            self.region_root = os.path.join(self.data_root, 'test_2k_region')
            self.data_root = os.path.join(self.data_root, 'test_2k_540p')


        dirs = os.listdir(self.data_root)
        data_list = []
        for d in dirs:
            if d == '.DS_Store':
                continue
            img0 = os.path.join(self.data_root, d, 'frame1.jpg')
            img1 = os.path.join(self.data_root, d, 'frame3.jpg')
            # points14 = os.path.join(self.data_root, d, 'frame3.jpg')

            points14 = os.path.join(self.data_root, d, 'inter14.jpg')
            points12 = os.path.join(self.data_root, d, 'inter12.jpg')
            points34 = os.path.join(self.data_root, d, 'inter34.jpg')
            gt = os.path.join(self.data_root, d, 'frame2.jpg')

            result = os.path.join(self.result_dir, d, 'abregion12.png')
            print(result)

            region13 = os.path.join(self.region_root, d, 'guide_flo13.npy')
            region31 = os.path.join(self.region_root, d, 'guide_flo31.npy')

            # data_list.append([img0, img1, points14, points12, points34, gt, d])
            data_list.append([img0, img1, points14, points12, points34, gt, d, region13, region31, result])

        self.data_list = data_list

        if self.training:
            self.transforms = transforms.Compose([
                # transforms.RandomCrop(228),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, index):

        imgpaths = [self.data_list[index][0], self.data_list[index][1], self.data_list[index][2], self.data_list[index][3], self.data_list[index][4], self.data_list[index][5], self.data_list[index][9]]
        # Load images
        images = [Image.open(pth) for pth in imgpaths]
        print(images)
        ## Select only relevant inputs
        # inputs = [int(e)-1 for e in list(self.inputs)]
        # inputs = inputs[:len(inputs)//2] + [3] + inputs[len(inputs)//2:]
        # images = [images[i] for i in inputs]
        # imgpaths = [imgpaths[i] for i in inputs]
        # Data augmentation
        size = (384, 192)
        flow13 = np.load(self.data_list[index][7]).astype(np.float32)
        flow31 = np.load(self.data_list[index][8]).astype(np.float32)
        flow = [flow13, flow31]
        if self.training:
            seed = random.randint(0, 2**32)
            images_ = []
            for img_ in images:
                img_ = img_.resize(size)
                random.seed(seed)
                images_.append(self.transforms(img_))
            images = images_

            gt = images[5]

            images = images[:5]

            return images, gt, flow
        else:
            T = self.transforms
            images = [T(img_.resize(size)) for img_ in images]

            gt = images[5]
            images = images[:5]
            imgpath = self.data_list[index][6]
            result = images[6]
            return images, gt, imgpath, flow, result

    def __len__(self):
        if self.training:
            return len(self.data_list)
        else:
            return len(self.data_list)

def get_loader(mode, data_root, batch_size, shuffle, num_workers, test_mode=None):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = ATD12k(data_root, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":

    dataset = ATD12k("/Users/shenjiaming/Documents/2022 S1/ELEC4712/MySuperGlue/atd12k_points", is_training=False)
    # print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=32, pin_memory=True)