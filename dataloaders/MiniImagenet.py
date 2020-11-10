import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
from .fewShotLearning import FewShotDataset


class MiniImagenet(FewShotDataset):
    def __init__(self, root, phase, n_way, k_spt, k_query,imgsz):
        super(MiniImagenet,self).__init__(root,phase, n_way, k_spt, k_query)
        self.imgsz = imgsz
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.imgsz, self.imgsz)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        self.path = os.path.join(self.root, 'images')  # image path
        csvdata = self.loadCSV(os.path.join(self.root, self.phase + '.csv'))  # csv path
        for i,(k,v) in enumerate(csvdata.items()):
            if i<1:
                print(k,len(v))
                print('\n')
        self.data = []
        self.img2label = {}
    
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [[img1, img2, ...], [img111, ...]]. Append list of imgs corresponding to a class.
            self.img2label[k] = i  # Shifting range of labels from (x,y) to (0,y-x). Needed for each phase:train,val,test.
            # print(k,self.img2label[k])
        self.cls_num = len(self.data)
        self.cls_items = [len(x) for x in self.data]

        self.get_all_items()

    def get_all_items(self):

        ######### for multiple items change self.support_x_batch into empty list and add support_x vecs. Also change __getitem__ accordingly
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch

        if self.phase=='train':
            range_val = 200000
        elif self.phase=='val':
            range_val = sum(self.cls_items)
        else:
            range_val = 600
        # 1.select n_way classes randomly
        for b in range(range_val):

            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            # for b in range(batchsz):                    # use this for multiple batches before using dataloader 
            # get indices for spt and qry images and labels
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_spt + self.k_qry, False)
                selected_imgs_idx = list(selected_imgs_idx)
                random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_spt])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_spt:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

        print('create batch',len(self.support_x_batch),len(self.support_x_batch[0]))

    def __getitem__(self,index):

        # init tensors for getting item
        support_x = torch.FloatTensor(self.k_spt*self.n_way, 3, self.imgsz, self.imgsz)
        support_y = np.zeros((self.k_spt*self.n_way), dtype=np.int)
        query_x = torch.FloatTensor(self.k_qry*self.n_way, 3, self.imgsz, self.imgsz)
        query_y = np.zeros((self.k_qry*self.n_way), dtype=np.int)

        flatten_support_x = [os.path.join(self.path, item)
                             for sublist in self.support_x_batch[index] for item in sublist]
        support_y = np.array(
            [self.img2label[item[:9]]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
             for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

        flatten_query_x = [os.path.join(self.path, str(item))
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.img2label[item[:9]]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.k_spt*self.n_way)
        query_y_relative = np.zeros(self.k_qry*self.n_way)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)


    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def __len__(self):
        return sum(self.cls_items)


