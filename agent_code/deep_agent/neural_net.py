from typing import Any
import torch
from torch import nn
from torchvision.transforms import ToTensor

import os

from .constants import ACTIONS

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(176 * 5, 512, dtype=torch.double)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 512, dtype=torch.double)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(512, len(ACTIONS), dtype=torch.double)

    def forward(self, x):
        #x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
    
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path

class Trainer:
    def __init__(self, model_file_name, loss_fn, optimizer_constructor, gamma, main_model_update = 4, target_model_update = 100) -> None:
        self.model_file_name = model_file_name
        self.main_model = NeuralNetwork()
        self.target_model = NeuralNetwork()

        if os.path.isfile(model_file_name):
            self.main_model.load_state_dict(torch.load(model_file_name))
            self.target_model.load_state_dict(torch.load(model_file_name))

        self.loss_fn = loss_fn
        self.optimizer = optimizer_constructor(self.main_model.parameters())
        self.gamma = gamma
        self.main_model_update = main_model_update
        self.target_model_update = target_model_update
        self.counter = 0
        self.last_pred = None

        self.main_model.train()
        #self.target_model.test()

    def forward(self, features):
        self.last_pred = self.main_model.forward(features)
        return self.last_pred

    def update(self, new_features, last_action, reward):
        with torch.no_grad():
            target_pred = self.target_model.forward(new_features)

        y = torch.clone(self.last_pred)
        y[last_action] = reward + self.gamma * torch.max(target_pred)

        loss = self.loss_fn(self.last_pred, y)
        loss.backward()

        self.counter += 1
        if self.counter % self.main_model_update == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.counter % self.target_model_update == 0:
            self.target_model.load_state_dict(self.main_model.state_dict())
            torch.save(self.target_model.state_dict(), self.model_file_name)
            torch.save(self.target_model.state_dict(), uniquify(self.model_file_name))
