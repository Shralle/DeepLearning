from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2
import random
import datasets.edge_utils as edge
from datasets.background_apply import apply_background
from utils.classes import *

from PIL import Image
from PIL import ImageChops
import numpy as np
from skimage.filters import sobel
from skimage.segmentation import watershed
import random

classes = {'front_door': (20, 100, 20),
           'back_door': (250, 250, 10),
           'fender': (20, 20, 250),
           'frame': (10, 250, 250),
           'bumper': (250, 10, 250),
           'hood': (250, 150, 10),
           'back_bumper': (150, 10, 150),
           'trunk': (10, 250, 10)}


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def apply_background(img, trg, land, pts=100):

    size = img.size[0]
    img_np = np.asarray(img)

    # Start by subtracting the two images
    sub = ImageChops.subtract(trg, img)

    # Making a mask
    sub_mask = Image.eval(sub, lambda a: a if a >= 20 else 0)

    # Creating numpy mask
    mask = np.zeros((size, size, 3))
    for i, p in enumerate(sub_mask.getdata()):
        if p == (0, 0, 0):
            mask[i // size, i % size, :] = [1, 1, 1]
    mask = mask.astype(bool)

    rmin, rmax, cmin, cmax = bbox(~mask)

    masked_array = np.ma.masked_array(data=img_np, mask=~mask, fill_value=0)

    pre_img = masked_array.filled()

    gray = np.asarray(Image.fromarray(pre_img).convert('L'))

    elevation_map = sobel(gray)

    markers = np.zeros_like(gray)
    markers[gray <= 50] = 1

    num_pts = 0
    # picking random points outside car:
    while num_pts < pts:
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        if not ((x >= rmin) and (x <= rmax) and (y >= cmin) and (y <= cmax)):
            markers[x, y] = 2
            num_pts += 1

    segmentation = watershed(elevation_map, markers)

    final = np.zeros((size, size, 3))

    for i, p in enumerate(land.getdata()):
        x, y = i // size, i % size
        if segmentation[x, y] == 1:
            final[x, y, :] = img_np[x, y, :]
        elif segmentation[x, y] == 2:
            final[x, y, :] = p

    return Image.fromarray(final.astype(np.uint8))

def find_class(pix, classes=classes, thres=50):

    for i, c in enumerate(classes):
        v = classes[c]
        d = np.sqrt((pix[0] - v[0])**2 + (pix[1] - v[1])**2 + (pix[2] - v[2])**2)
        if d < thres:
            return i

    return None

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):

    image = np.array(image)
    image = image[:, :, ::-1].copy()

    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    image = image[:, :, ::-1].copy()

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):

    image = np.array(image)
    mask = np.array(mask)
    image = image[:, :, ::-1].copy()
    mask = mask[:, :, ::-1].copy()

    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(0, 0, 0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(0, 0, 0,))
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)

    image = image[:, :, ::-1].copy()
    mask = mask[:, :, ::-1].copy()

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):

    image = np.array(image)
    mask = np.array(mask)
    image = image[:, :, ::-1].copy()
    mask = mask[:, :, ::-1].copy()

    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    image = image[:, :, ::-1].copy()
    mask = mask[:, :, ::-1].copy()

    return image, mask

def randomZoom(image, mask, zoom_limit=0.25, u=0.5):

    image = np.array(image)
    mask = np.array(mask)
    image = image[:, :, ::-1].copy()
    mask = mask[:, :, ::-1].copy()

    if np.random.random() < u:
        value = np.random.uniform(zoom_limit, 1)
        h, w = image.shape[:2]
        h_taken = int(value * h)
        w_taken = int(value * w)
        h_start = np.random.randint(0, h - h_taken)
        w_start = np.random.randint(0, w - w_taken)
        image = image[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
        mask = mask[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
        image = cv2.resize(image, (h, w), cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (h, w), cv2.INTER_CUBIC)

    image = image[:, :, ::-1].copy()
    mask = mask[:, :, ::-1].copy()

    return image, mask

def preprocess(pil_img, size, land=None, classes=classes, source=1, u=0.1, thres=50, prefix='../data/'):

    if source == 1:

        w, h = pil_img.size

        trg = pil_img.crop([0, 0, w // 2, h])
        img = pil_img.crop([w // 2, 0, w, h])

        trg = trg.resize((size, size))
        img = img.resize((size, size))

    elif source == 2:

        img = Image.open(prefix + 'opel/Astra_no_segments_Camera_a-{}.png'.format(pil_img))
        trg = Image.open(prefix + 'opel/Astra_w_segments_Camera_a-{}.png'.format(pil_img))

        trg = trg.resize((size, size))
        img = img.resize((size, size))
        
        # Applying random background to img
        if land is not None:
            img = apply_background(img, trg, land)

    elif source == 3:

        img = Image.open(prefix + 'clio/clio_no_segments_Editor_a{}.png'.format(pil_img))
        trg = Image.open(prefix + 'clio/clio_w_segments_Editor_a{}.png'.format(pil_img))

        trg = trg.resize((size, size))
        img = img.resize((size, size))
        
        # Applying random background to img
        if land is not None:
            img = apply_background(img, trg, land)

    elif source == 4:

        img = Image.open(prefix + 'clio_street/clio_street_no_segments-{}.png'.format(pil_img))
        trg = Image.open(prefix + 'clio_street/clio_street-{}.png'.format(pil_img))

        trg = trg.resize((size, size))
        img = img.resize((size, size))

    elif source == 5:

        img = Image.open(prefix + '5doors/sti_no_segs_i2-{}.png'.format(pil_img))
        trg = Image.open(prefix + '5doors/sti_no_segs_black_i1_i2-{}.png'.format(pil_img))

        trg = trg.resize((size, size))
        img = img.resize((size, size))
        
        # Applying random background to img
        if land is not None:
            img = apply_background(img, trg, land)

    elif source == 0:

        img = pil_img.resize((size, size))
        trg = pil_img.copy()

    else:
        print('NOT IMPLEMENTED SOURCE {}'.format(source))
        return

    # Apply random transforms

    img_nd, trg_nd = randomHorizontalFlip(img, trg, u=u)

    img_nd, trg_nd = randomShiftScaleRotate(img_nd, trg_nd,
                                            shift_limit=(-0.1, 0.1),
                                            scale_limit=(-0.1, 0.1),
                                            aspect_limit=(-0.1, 0.1),
                                            rotate_limit=(-20, 20),
                                            u=u)

    img_nd = randomHueSaturationValue(img_nd,
                                      hue_shift_limit=(-15, 15),
                                      sat_shift_limit=(-5, 5),
                                      val_shift_limit=(-15, 15),
                                      u=u)

    img_nd, trg_nd = randomZoom(img_nd, trg_nd,
                                zoom_limit=0.25,
                                u=u)

    if source >= 1:

        trg = Image.fromarray(trg_nd)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        mask = np.zeros((1, size, size))
        target = np.zeros((len(classes) + 1, size, size))

        for i, p in enumerate(trg.getdata()):
            cl = find_class(p, classes, thres=thres)
            if source == 2 or source == 3 or source == 4:
                if cl == 1:
                    cl = None
            if cl is not None:
                mask[0, i // size, i % size] = cl + 1
                target[cl + 1, i // size, i % size] = 1
            else:
                target[0, i // size, i % size] = 1

       # _edgemap = mask[0]
       # _edgemap = edge.mask_to_onehot(_edgemap, len(classes) + 1)
        #_edgemap = edge.onehot_to_binary_edges(_edgemap, 2, len(classes) + 1)
        #edgemap = torch.from_numpy(_edgemap).float()

        # Calculating edgemaps
        _edge_maps = edge.onehot_to_binary_edges(target, 2, len(classes) + 1)
        edgemap = torch.from_numpy(_edge_maps).float()

    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    trg_trans = trg_nd.transpose((2, 0, 1))
    if img_trans.max() > 1:
        img_trans = img_trans / 255
    if trg_trans.max() > 1:
        trg_trans = trg_trans / 255

    img_inp = np.zeros_like(img_trans)

    img_inp[0, :, :] = img_trans[0, :, :] - 0.485 / 0.229
    img_inp[1, :, :] = img_trans[1, :, :] - 0.456 / 0.224
    img_inp[2, :, :] = img_trans[2, :, :] - 0.406 / 0.225

    if source >= 1:
        return img_inp, mask, target, trg_trans, img_trans, edgemap
    else:
        return img_inp, None, None, trg_trans, img_trans, None
        
class BasicDataset(Dataset):
    def __init__(self, imgs_dir, size=256, opel_dir='../data/opel/', clio_dir='../data/clio/', street_dir='../data/clio_street/', door_dir='../data/5doors/', opel_pre='Astra_w_segments_Camera_a-', clio_pre='clio_w_segments_Editor_a', street_pre='clio_street-', door_pre='sti_no_segs_i2-', land_pre='data/landscape/images/', frac=0.5, seed=0):
        self.imgs_dir = imgs_dir
        self.opel_imgs_dir = opel_dir
        self.opel_pre = opel_pre
        self.clio_imgs_dir = clio_dir
        self.clio_pre = clio_pre
        self.street_imgs_dir = street_dir
        self.street_pre = street_pre
        self.door_imgs_dir = door_dir
        self.door_pre = door_pre
        self.land_pre = land_pre
        self.size = size

        self.lands = [splitext(file)[0] for file in listdir(land_pre)
                    if not file.startswith('.')]
        
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

        self.opel_ids = ['OPEL_' + splitext(file)[0][-4:] for file in listdir(opel_dir)
                         if not file.startswith('.') and 'w' in file]

        self.clio_ids = ['CLIO_' + splitext(file)[0][-4:] for file in listdir(clio_dir)
                         if not file.startswith('.') and 'w' in file]

        self.door_ids = ['DOOR_' + splitext(file)[0][-4:] for file in listdir(door_dir)
                         if not file.startswith('.') and 'sti_no_segs_black_i1_i2-' in file]

        # street images
        self.street_ids = ['STRT_' + splitext(file)[0][-4:] for file in listdir(street_dir)
                         if not file.startswith('.') and 'no' in file]

        self.num_ids = int(frac * len(self.ids))

        # Randomizing lists
        random.seed(seed)

        random.shuffle(self.opel_ids)
        random.shuffle(self.clio_ids)
        random.shuffle(self.door_ids)
        random.shuffle(self.street_ids)

        self.ids.extend(self.opel_ids[:self.num_ids // 2])
        #self.ids.extend(self.clio_ids[:self.num_ids // 8])
        self.ids.extend(self.door_ids[:self.num_ids // 2])
        #self.ids.extend(self.street_ids[:self.num_ids // 8])

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        
        land = glob(self.land_pre + self.lands[random.randint(0, len(self.lands)-1)] + '.jpg')
        land = Image.open(land[0])
        land = land.convert('RGB')
        land = land.resize((self.size, self.size))
        
        if idx[:4] == 'OPEL':
            source = 2
            img_file = glob(self.opel_imgs_dir + self.opel_pre + idx[5:] + '.png')
            assert len(img_file) == 1, \
                f'Either no image or multiple images found for the ID {idx}: {img_file}'
            img = idx[5:]
        elif idx[:4] == 'CLIO':
            source = 3
            img_file = glob(self.clio_imgs_dir + self.clio_pre + idx[5:] + '.png')
            assert len(img_file) == 1, \
                f'Either no image or multiple images found for the ID {idx}: {img_file}'
            img = idx[5:]
        elif idx[:4] == 'STRT':
            source = 4
            img_file = glob(self.street_imgs_dir + self.street_pre + idx[5:] + '.png')
            assert len(img_file) == 1, \
                f'Either no image or multiple images found for the ID {idx}: {img_file}'
            img = idx[5:]
        elif idx[:4] == 'DOOR':
            source = 5
            img_file = glob(self.door_imgs_dir + self.door_pre + idx[5:] + '.png')
            assert len(img_file) == 1, \
                f'Either no image or multiple images found for the ID {idx}: {img_file}'
            img = idx[5:]
        else:
            source = 1
            img_file = glob(self.imgs_dir + idx + '.jpg')
            assert len(img_file) == 1, \
                f'Either no image or multiple images found for the ID {idx}: {img_file}'
            img = Image.open(img_file[0])
            land = None

        img, mask, target, trg, rawimg, edge = preprocess(img, self.size, land, classes, u=0.5, source=source)

        return {'image': torch.from_numpy(img).type(torch.FloatTensor), 'mask': torch.from_numpy(mask).squeeze(0).type(torch.FloatTensor), 'target': torch.from_numpy(trg).type(torch.FloatTensor), 'target_mask': torch.from_numpy(target).type(torch.FloatTensor), 'raw_image': torch.from_numpy(rawimg).type(torch.FloatTensor), 'index': idx, 'edge': edge.type(torch.FloatTensor)}

        print("Initializing Datasets and Dataloaders...")

dir_img = '../carseg_data/save/'
val_percent=0.1

dataset = BasicDataset(dir_img)
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=False)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
