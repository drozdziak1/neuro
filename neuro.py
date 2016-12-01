#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class NeuralNet(object):

    def __init__(self):
        # hyper parameters
        self.inputLayers = 2
        self.outputLayers = 1
        self.hiddenLayers = 3

        # weights
        self.W1 = np.random.randn(self.inputLayers, self.hiddenLayers)

        self.W2 = np.random.randn(self.hiddenLayers, self.outputLayers)

    def forward(self, X):
        # input propagation
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        # the activation function
        return 1 / (1 + np.exp(-z))

NN = NeuralNet()

yHat = NN.forward([[8, 8], [8, 8]])

print(yHat)
