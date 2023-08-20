#!/usr/bin/env python3
import numpy as np
from teenygrad.tensor import Tensor
from tqdm import trange
import gzip, os

# sorted in order of increasing complexity
from typing import List
from teenygrad.helpers import dedup, getenv

class Optimizer:
  def __init__(self, params: List[Tensor], lr: float):
    # if it's None, but being put into an optimizer, set it to True
    for x in params:
      if x.requires_grad is None: x.requires_grad = True

    self.params: List[Tensor] = dedup([x for x in params if x.requires_grad])
    self.buffers: List[Tensor] = dedup([x for x in params if not x.requires_grad])   # buffers are still realized
    self.lr = Tensor([lr], requires_grad=False).contiguous()

  def zero_grad(self):
    for param in self.params: param.grad = None

  def realize(self, extra=None):
    # TODO: corealize
    # NOTE: in extra is too late for most of the params due to issues with assign
    for p in extra + self.params + self.buffers if extra is not None else self.params + self.buffers:
      p.realize()

class SGD(Optimizer):
  def __init__(self, params: List[Tensor], lr=0.001, momentum=0, weight_decay=0.0, nesterov=False):
    super().__init__(params, lr)
    self.momentum, self.wd, self.nesterov = momentum, weight_decay, nesterov
    self.b = [Tensor.zeros(*t.shape, device=t.device, requires_grad=False) for t in self.params] if self.momentum else []

  # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
  def step(self) -> None:
    for i, t in enumerate(self.params):
      assert t.grad is not None
      g = t.grad.realize() + self.wd * t.detach()
      if self.momentum:
        self.b[i].assign(self.momentum * self.b[i] + g).realize()  # NOTE: self.b[i] is zero on the first run, no if required
        g = (g + self.momentum * self.b[i]) if self.nesterov else self.b[i]
      t.assign(t.detach() - g * self.lr)
    self.realize(self.b)


def sparse_categorical_crossentropy(out, Y):
  num_classes = out.shape[-1]
  YY = Y.flatten().astype(np.int32)
  y = np.zeros((YY.shape[0], num_classes), np.float32)
  # correct loss for NLL, torch NLL loss returns one per row
  y[range(y.shape[0]),YY] = -1.0*num_classes
  y = y.reshape(list(Y.shape)+[num_classes])
  y = Tensor(y)
  return out.mul(y).mean()

def train(model, X_train, Y_train, optim, steps, BS=128, lossfn=sparse_categorical_crossentropy,
        transform=lambda x: x, target_transform=lambda x: x, noloss=False):
  Tensor.training = True
  losses, accuracies = [], []
  for i in (t := trange(steps, disable=getenv('CI', False))):
    samp = np.random.randint(0, X_train.shape[0], size=(BS))
    x = Tensor(transform(X_train[samp]), requires_grad=False)
    y = target_transform(Y_train[samp])

    # network
    out = model.forward(x) if hasattr(model, 'forward') else model(x)

    loss = lossfn(out, y)
    optim.zero_grad()
    loss.backward()
    if noloss: del loss
    optim.step()

    # printing
    if not noloss:
      cat = np.argmax(out.cpu().numpy(), axis=-1)
      accuracy = (cat == y).mean()

      loss = loss.detach().cpu().numpy()
      losses.append(loss)
      accuracies.append(accuracy)
      t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))
  return [losses, accuracies]


def fetch_mnist():
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  BASE = os.path.dirname(__file__)+"/../tinygrad/extra/datasets"
  X_train = parse(BASE+"/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_train = parse(BASE+"/mnist/train-labels-idx1-ubyte.gz")[8:]
  X_test = parse(BASE+"/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_test = parse(BASE+"/mnist/t10k-labels-idx1-ubyte.gz")[8:]
  return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = fetch_mnist()

# create a model with a conv layer
class TinyConvNet:
  def __init__(self):
    # https://keras.io/examples/vision/mnist_convnet/
    conv = 3
    #inter_chan, out_chan = 32, 64
    inter_chan, out_chan = 8, 16   # for speed
    self.c1 = Tensor.scaled_uniform(inter_chan,1,conv,conv)
    self.c2 = Tensor.scaled_uniform(out_chan,inter_chan,conv,conv)
    self.l1 = Tensor.scaled_uniform(out_chan*5*5, 10)

  def forward(self, x:Tensor):
    x = x.reshape(shape=(-1, 1, 28, 28)) # hacks
    x = x.conv2d(self.c1).relu().max_pool2d()
    x = x.conv2d(self.c2).relu().max_pool2d()
    x = x.reshape(shape=[x.shape[0], -1])
    return x.dot(self.l1).log_softmax()

if __name__ == "__main__":
  np.random.seed(1337)
  model = TinyConvNet()
  optimizer = SGD([model.c1, model.c2, model.l1], lr=0.001)
  train(model, X_train, Y_train, optimizer, steps=100)
  assert evaluate(model, X_test, Y_test) > 0.93   # torch gets 0.9415 sometimes