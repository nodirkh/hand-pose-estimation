from model import *
import torch
import numpy as np
import torchvision
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import math
import torch.nn as nn
import torch.optim as optim

from glob import glob
import pandas
import matplotlib.pyplot as plt

device = 'cuda:0'
num_joints = 21

class Dataset(Dataset):
    def __init__(self, method=None):
        self.root = '/path/to/dataset'
        self.x_data = []
        self.y_data = []
        if method == 'train':
            self.root = self.root + 'train/'
            self.img_path = sorted(glob(self.root + 'rgb/*.jpg'))

        elif method == 'test':
            self.root = self.root + 'test/'
            self.img_path = sorted(glob(self.root + 'rgb/*.jpg'))

        for i in tqdm.tqdm(range(len(self.img_path))):
            img = cv2.imread(self.img_path[i], cv2.IMREAD_COLOR)
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            self.x_data.append(img)

            num = self.img_path[i].split('.')[0].split('/')[-1]
            img_pkl = self.root + 'meta/' + str(num) + '.pkl'
            pkl = pandas.read_pickle(img_pkl)
            coords_2d = pkl['coords_2d']
            self.y_data.append(coords_2d)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        transform1 = torchvision.transforms.ToTensor()
        new_x_data = transform1(self.x_data[idx])

        return new_x_data, self.y_data[idx]


class Tester(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self._build_model()

        self.root = '/path/to/dataset'

        weight_PATH = '/path/to/weight'
        self.poseNet.load_state_dict(torch.load(weight_PATH))

        print("Testing...")

    def _build_model(self):
        # 2d pose estimator
        poseNet = CPM2DPose()
        self.poseNet = poseNet.to(device)

        print('Finish build model.')

    def heatmap2skeleton(self, heatmapsPoseNet):
        skeletons = np.zeros((heatmapsPoseNet.shape[0], heatmapsPoseNet.shape[1], 2))
        for m in range(heatmapsPoseNet.shape[0]):
            for i in range(heatmapsPoseNet.shape[1]):
                u, v = np.unravel_index(np.argmax(heatmapsPoseNet[m][i]), (32, 32))
                skeletons[m, i, 0] = u * 8
                skeletons[m, i, 1] = v * 8
        return skeletons

    @staticmethod
    def plot_hand(coords_hw, axis, color_fixed=None, linewidth='1'):
        colors = np.array([[0., 0., 1.],
                           [0., 0., 1.],
                           [0., 0., 1.],
                           [0., 0., 1.],
                           [1., 0., 1.],
                           [1., 0., 1.],
                           [1., 0., 1.],
                           [1., 0., 1.],
                           [1., 0., 0.],
                           [1., 0., 0.],
                           [1., 0., 0.],
                           [1., 0., 0.],
                           [0., 1., 0.],
                           [0., 1., 0.],
                           [0., 1., 0.],
                           [0., 1., 0.],
                           [1., 0.5, 0.],
                           [1., 0.5, 0.],
                           [1., 0.5, 0.],
                           [1., 0.5, 0.]])
        bones = [((1, 0), colors[0, :]),
                 ((2, 1), colors[1, :]),
                 ((3, 2), colors[2, :]),
                 ((4, 3), colors[3, :]),
                 ((0, 5), colors[4, :]),
                 ((5, 6), colors[5, :]),
                 ((6, 7), colors[6, :]),
                 ((7, 8), colors[7, :]),
                 ((0, 9), colors[8, :]),
                 ((9, 10), colors[9, :]),
                 ((10, 11), colors[10, :]),
                 ((11, 12), colors[11, :]),
                 ((0, 13), colors[12, :]),
                 ((13, 14), colors[13, :]),
                 ((14, 15), colors[14, :]),
                 ((15, 16), colors[15, :]),
                 ((0, 17), colors[16, :]),
                 ((17, 18), colors[17, :]),
                 ((18, 19), colors[18, :]),
                 ((19, 20), colors[19, :])]
        for connection, color in bones:
            coord1 = coords_hw[connection[0], :]
            coord2 = coords_hw[connection[1], :]
            coords = np.stack([coord1, coord2])
            if color_fixed is None:
                axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
            else:
                axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)

    def test(self):
        path = self.root + 'test/rgb/106.jpg'
        pkl_path = self.root + 'test/meta/106.pkl'
        print(path)
        print(pkl_path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        transform1 = torchvision.transforms.ToTensor()
        x_data = transform1(img)
        pkl = pandas.read_pickle(pkl_path)
        y_test = pkl['coords_2d']

        self.poseNet.eval()

        print("Testing 106.jpg heatmap")

        heatmapsPoseNet = self.poseNet(x_data.unsqueeze(0).cuda()).cpu().detach().numpy()
        heatmap = np.transpose(heatmapsPoseNet[0], (1, 2, 0))

        plt.rcParams['figure.figsize'] = [12, 5]
        for i in range(num_joints):
            ax = plt.subplot(3, 7, i + 1)
            ax.imshow(heatmap[:, :, i])
        plt.savefig('heatmap.png')
        plt.show()

        print("Testing 106.jpg heatmap")

        ax = plt.subplot(1, 1, 1)
        ax.imshow(img)
        self.plot_hand(self.heatmap2skeleton(heatmapsPoseNet)[0], ax)
        plt.savefig('results.png') 
        plt.show()

        print(f"Object size: {self.heatmap2skeleton(heatmapsPoseNet).shape}")

    def calc_err_avg(self):
      dataset = Dataset(method='test') # load dataset in test mode
      dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
      total_error = 0
      self.poseNet.eval()
      for x_data, y_data in dataloader: # tuple destructor
        # print(x_data.shape)
        # print(y_data.shape)
        # print(type(y_data))
        heatmapsPoseNet = self.poseNet(x_data.cuda()).cpu().detach().numpy()
        x_pred = self.heatmap2skeleton(heatmapsPoseNet)
        # print(x_pred.shape)
        # print(x_pred.shape)
        total_error += self.calc_error(x_pred, y_data)
        
      with open('error.txt', 'w') as error_file:
        error_file.write(str((total_error / 500).cpu().detach().numpy()))
      print("Average error : ", (total_error / 500).cpu().detach().numpy())
          

    
    def calc_error(self, predicted, ground_truth):
      """
      Calculates error per image
      Averages the pixel-joint error per joint
      """
      err = 0
      k = num_joints
      for i in range(predicted.shape[0]):
        for j in range(k):
          j_ix_true = ground_truth[i][j][0]
          j_ix_pred = predicted[i][j][0]

          j_iy_true = ground_truth[i][j][1]
          j_iy_pred = predicted[i][j][1]

          lhs = (j_ix_pred - j_ix_true)**2
          rhs = (j_iy_pred - j_iy_true)**2
          err += np.sqrt(lhs+rhs)

      err /= k

      return err

class Trainer(object):
    def __init__(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self._build_model()

        self.cost = nn.MSELoss()
        self.optimizer = optim.Adam(self.poseNet.parameters(), lr=self.learning_rate)

        dataset = Dataset(method='train')
        self.root = dataset.root
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Load of pretrained_weight file
        weight_PATH = '/path/to/weight'
        self.poseNet.load_state_dict(torch.load(weight_PATH))

    def _build_model(self):
        # 2d pose estimator
        poseNet = CPM2DPose()
        self.poseNet = poseNet.to(device)
        self.poseNet.train()

    def skeleton2heatmap(self, _heatmap, keypoint_targets):
        heatmap_gt = torch.zeros_like(_heatmap, device=_heatmap.device)

        keypoint_targets = (((keypoint_targets)) // 8)
        for i in range(keypoint_targets.shape[0]):
            for j in range(21):
                x = int(keypoint_targets[i, j, 0])
                y = int(keypoint_targets[i, j, 1])
                heatmap_gt[i, j, x, y] = 1

        heatmap_gt = heatmap_gt.detach().cpu().numpy()
        for i in range(keypoint_targets.shape[0]):
            for j in range(21):
                heatmap_gt[i, j, :, :] = cv2.GaussianBlur(heatmap_gt[i, j, :, :], ksize=(3, 3), sigmaX=2,
                                                          sigmaY=2) * 9 / 1.1772
        heatmap_gt = torch.FloatTensor(heatmap_gt).to(device)
        return heatmap_gt

    def train(self):
        date = '202211'
        current_best = np.inf # keep track of the best epoch
        for epoch in tqdm.tqdm(range(self.epochs + 1)):
            if epoch % 2 == 0:
                torch.save(self.poseNet.state_dict(), "_".join(['/content/model_', str(epoch), '.pth']))
            
            train_loss = 0
            for batch_idx, samples in enumerate(self.dataloader):
                x_data, y_data = samples # load the data
                heatmapsPoseNet = self.poseNet(x_data.cuda())
                y_true = self.skeleton2heatmap(heatmapsPoseNet, y_data) # get gt values

                ## perform training
                self.optimizer.zero_grad()
                loss = self.cost(heatmapsPoseNet, y_true)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                ## Write train result
                if batch_idx % 20 == 0:
                    with open('train_result_' + date + '.txt', 'a') as f:
                        f.write('Epoch {:4d}/{} Batch {}/{}\n'.format(
                            epoch, self.epochs, batch_idx, len(self.dataloader)
                        ))
                    print('Epoch {:4d}/{} Batch {}/{}'.format(
                        epoch, self.epochs, batch_idx, len(self.dataloader)
                    ))
            ## update loss
            if train_loss < current_best:
              current_best = train_loss
              torch.save(self.poseNet.state_dict(), "weight.pth")

        print('Finish training.')


def main():
    epochs = 20
    batchSize = 16
    learningRate = 1e-5

    trainer = Trainer(epochs, batchSize, learningRate)
    trainer.train()

    tester = Tester(batchSize)
    tester.test()
    tester.calc_err_avg() # function to calculate the average error of 500 imgs

if __name__ == '__main__':
    main()