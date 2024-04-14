import torch
import torch.nn as nn
import torch.nn.functional as F 


class PhysioNet(nn.Module):
    def __init__(self):
        super(PhysioNet, self).__init__()
        #input shape 34 * 8
        #output shape 16 * 8
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride = 1)
        #input shape 16 * 8
        #output shape 32 * 8
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride = 1)
        #input shape 32 * 8
        #output shape 16 * 4
        self.pool1 = nn.MaxPool2d(2,2)
        #input shape 16 * 4
        #output shape 32 * 4
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride = 1)
        self.fc1 = nn.Linear(64*17*2, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x, mode):
        #print("In PhysioNet")
        x = x.unsqueeze(1)
        print(x.shape)
        x = F.leaky_relu(self.conv1(x))
        print(x.shape)
        x = F.leaky_relu(self.conv2(x))
        print(x.shape)
        x = self.pool1(x)
        print(x.shape)
        x = F.leaky_relu(self.conv3(x))
        print(x.shape)
        x = x.view(-1, 64*17*2)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        print(x.shape)
        x = F.leaky_relu(self.fc2(x))
        print(x.shape)
        x = F.leaky_relu(self.fc3(x))
        print(x.shape)
        if mode != "both":
            x = self.fc4(x)
        return x

class NotesNet(nn.Module):
    def __init__(self):
        super(NotesNet, self).__init__()
        #input shape 1 * 80 * 768
        #output shape 16 * 40 * 384
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2, stride = 2)
        #input shape 16 * 40 * 384
        #output shape 16 * 20 * 192
        self.pool2 = nn.MaxPool2d(2,2)
        #input shape 16 * 20 * 192
        #output shape 32 * 20 * 192
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride = 1)
        #input shape 32 * 20 * 192
        #output shape 64 * 10 * 96
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride = 2)
        self.fc1 = nn.Linear(64*10*96, 2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x, mode):
        #input is of shape (batch_size=32, 3, 224, 224) if you did the dataloader right
        x = x.unsqueeze(1)
        #print(x.shape)
        x = x.to(torch.float32)
        #print(x.dtype)
        x = F.leaky_relu(self.conv1(x))
        #print(x.shape)
        x = self.pool2(x)
        #print(x.shape)
        x = F.leaky_relu(self.conv3(x))
        #print(x.shape)
        x = F.leaky_relu(self.conv4(x))
        #print(x.shape)
        x = x.view(-1, 64*10*96)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        #print(x.shape)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        #print(x.shape)
        x = F.leaky_relu(self.fc3(x))
        if mode != "both":
            x = self.fc4(x)
        return x


class EpisodeNet(nn.Module):
    def __init__(self, mode = "both"):
        super(EpisodeNet, self).__init__()
        self.mode = mode
        self.physio_net = PhysioNet()
        self.notes_net = NotesNet()
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, physio_x, notes_x):
        if self.mode == "both":
            physio_x = self.physio_net(physio_x, self.mode)
            #print("physio_x.shape", physio_x.shape)
            notes_x = self.notes_net(notes_x, self.mode)
            #print("notes_x.shape", notes_x.shape)
            x = torch.cat((physio_x, notes_x), axis = -1)
            #print("notes_x.shape", notes_x.shape)
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
        elif self.mode == "physio":
            x = self.physio_net(physio_x, self.mode)
        else:
            x = self.notes_net(notes_x, self.mode)
        return x