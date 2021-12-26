import json
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import numpy as np
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
#from torchvision import transforms as T
from scipy.ndimage import binary_erosion
import shutil
import math
from math import pi, cos, sin

import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def bw_on_image(im, bw, color):
    for i in range(0, im.shape[2]):
        t = im[:, :, i]
        t[bw] = color[i]
        im[:, :, i] = t
    #return img
    
def find_edge(bw, strel=5):
    return np.bitwise_xor(bw, binary_erosion(bw, structure=np.ones((strel, strel))).astype(bw.dtype))     

def getMaskfromPoly(imgsize,ypoints,xpoints):
    
    out=np.zeros((imgsize[0],imgsize[1]))
    coords=lst=[(i,j) for i,j in zip(xpoints,ypoints) ]
    poly=Polygon(coords)
     
    for i in range(imgsize[0]):
        for j in range(imgsize[1]):
            p1=Point(i,j)
            if poly.contains(p1):
                out[i,j]=1
    return out
        

class BrainTumorDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root,mode))))
        #print(self.imgs)
        f=open(root+'annotations_'+mode.lower()+'.json')
        self.annotations=json.load(f)
        self.mode=mode

        #xpoints=annotations['y0.jpg19127']['regions'][0]['shape_attributes']['all_points_x']
        #self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
    
    def getMaskfromPoly(self,imgname,imgsize):
        #print(imgname)

        dic=self.annotations[imgname+str(os.path.getsize(self.root+self.mode+'/'+imgname))]['regions'][0]['shape_attributes']
            #try:
        if  dic['name']=='polygon':
            xpoints=dic['all_points_x']
            ypoints=dic['all_points_y']

        elif dic['name']=='ellipse':
            u=dic['cx']       #x-position of the center
            v=dic['cy']      #y-position of the center
            a=dic['rx']       #radius on the x-axis
            b=dic['ry']      #radius on the y-axis
            t_rot=dic['theta'] #rotation angle
            t = np.linspace(0, 2*pi, 100)
            Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
                     #u,v removed to keep the same center location
            R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])  
                     #2-D rotation matrix
            Ell_rot = np.zeros((2,Ell.shape[1]))
            for i in range(Ell.shape[1]):
                Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
                xpoints=u+Ell_rot[0,:]
                ypoints=v+Ell_rot[1,:]
                
        elif dic['name']=='circle':
            theta = np.linspace( 0 , 2 * np.pi , 150 )
            radius = dic['r']
            xpoints = dic['cx']+radius * np.cos( theta )
            ypoints = dic['cy']+radius * np.sin( theta ) 
            
        out=np.zeros((imgsize[0],imgsize[1]))
        coords=lst=[(i,j) for i,j in zip(ypoints,xpoints) ]
        poly=Polygon(coords)

        for i in range(imgsize[0]):
            for j in range(imgsize[1]):
                p1=Point(i,j)
                if poly.contains(p1):
                    out[i,j]=1

        return out

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.mode+'/', self.imgs[idx])
        #mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        #mask = Image.open(mask_path)
        mask=self.getMaskfromPoly(self.imgs[idx],np.array(img).shape)
        img=img.resize((280,350),Image.NEAREST)
        mask=Image.fromarray(np.uint8(mask)).convert('RGB')
        mask=mask.resize((280,350),Image.NEAREST)
        mask = np.array(mask)
        mask=mask[:,:,0]
        
        #imt=np.array(img.copy())
        #self.bw_on_image(imt,(mask>0),[255,0,255])
        #plt.imshow(img)
        #plt.show()
        #plt.imshow(mask)
        #plt.show()
        
        # convert the PIL Image into a numpy array
        
        
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            #print(target)
            img, target = self.transforms(img, target)
            #print(target)
            #img=self.transforms(img)

        return img, target
    
    def bw_on_image(self,im, bw, color):
        for i in range(0, im.shape[2]):
            t = im[:, :, i]
            t[bw] = color[i]
            im[:, :, i] = t

    def __len__(self):
        return len(self.imgs)   
    


if __name__=='__main__':
    import sys
    import torchvision
    path=sys.argv[1]
    batch_size=4
    imgs = list(sorted(os.listdir(os.path.join(path,'TEST'))))
    ds_test=BrainTumorDataset(path,'TEST',get_transform(train=False))
    test_dataloader = DataLoader(ds_test, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load('maskRcnnModelStateDic_50epochs.pt', map_location='cpu'))
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    for images,targets in test_dataloader:
        images = list(img.to(cpu_device) for img in images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model.eval()
        model=model.to(cpu_device)
        outputs = model(images)
        targets2 = [{}] * len(images)
        for j in range(len(targets2)):
            for j2, v2 in targets.items():
                targets2[j][j2] = v2[j].to(cpu_device)
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        for i in range(len(outputs)):
            plt.imsave(path+'/'+'SG_'+imgs[targets2[0]['image_id'].item()]+'.png',np.squeeze(np.moveaxis(outputs[i]['masks'][0].detach().numpy(),0,-1))>0.5,format='png',cmap='gray')
        print('image '+imgs[targets2[0]['image_id'].item()]+' Converted')
	
	