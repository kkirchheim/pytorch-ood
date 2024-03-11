import unittest

import torch

from src.pytorch_ood.loss import VOSRegLoss
from tests.helpers.model import SegmentationModel

torch.manual_seed(123)


class TestEnergyRegularization(unittest.TestCase):
    """
    Test code for energy bounded learning
    """
    def init_loss(self,num_classes,reduction='mean',alpha=0.1):
        weights_energy = torch.nn.Linear(num_classes, 1).cpu()
        torch.nn.init.uniform_(weights_energy.weight)
        phi = torch.nn.Linear(1, 2).cpu()
        criterion = VOSRegLoss(phi,weights_energy,alpha=alpha, device='cpu' ,reduction=reduction)
        return criterion


    
    def test_forward(self):
        criterion = self.init_loss(10)
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_forward_only_positive(self):
        criterion = self.init_loss(10)
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_forward_only_negative(self):
        criterion = self.init_loss(10)
        logits = torch.randn(size=(128, 10))
        target = torch.ones(size=(128,)).long() * -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_set_alpha(self):
        criterion = self.init_loss(10,alpha=2)
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)
        
    def test_segmentation(self):
        model = SegmentationModel()
        criterion = self.init_loss(3,reduction='sum')
        x = torch.randn(size=(10, 3, 32, 32))
        target = torch.zeros(size=(10, 32, 32)).long()
        target[0, 0, 0] = -1

        logits = model(x)
        loss = criterion(logits, target)
        # self.assertEqual(loss.shape, (10, 32, 32))
        # None Loss geht nicht , weil unterschiedliche größe von loss und uncertain loss
        self.assertNotEqual(loss,0)      
        loss.mean().backward()

