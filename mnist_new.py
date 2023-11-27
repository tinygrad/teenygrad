#!/usr/bin/env python3
# To run on CLI -> "./mnist_py"
import os, gzip
import numpy as np

from teenygrad import Tensor
from teenygrad.nn import optim

def fetch_mnist(for_conv=True):
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  BASE = os.path.dirname(__file__)+"/extra/datasets"
  X_train = parse(BASE+"/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_train = parse(BASE+"/mnist/train-labels-idx1-ubyte.gz")[8:]
  X_test = parse(BASE+"/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_test = parse(BASE+"/mnist/t10k-labels-idx1-ubyte.gz")[8:]
  if for_conv:
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)
  return X_train, Y_train, X_test, Y_test

class TinyConvNet:
  def __init__(self):
    # https://keras.io/examples/vision/mnist_convnet/
    kernel_sz = 3
    in_chan, out_chan = 8, 16   # Reduced for speed
    self.c1 = Tensor.scaled_uniform(in_chan, 1, kernel_sz, kernel_sz)
    self.c2 = Tensor.scaled_uniform(out_chan, in_chan, kernel_sz, kernel_sz)
    self.l1 = Tensor.scaled_uniform(out_chan*5*5, 10)

  def __call__(self, x: Tensor):
    x = x.conv2d(self.c1).relu().max_pool2d()
    x = x.conv2d(self.c2).relu().max_pool2d()
    x = x.reshape(shape=[x.shape[0], -1])
    return x.dot(self.l1).log_softmax()

if __name__ == "__main__":
  NUM_STEPS = 100
  BS = 128
  LR = 0.001

  X_train, Y_train, X_test, Y_test = fetch_mnist()
  model = TinyConvNet()
  opt = optim.Adam([model.c1, model.c2, model.l1], lr=LR)

  with Tensor.train():
    for step in range(NUM_STEPS):
      samp = np.random.randint(0, X_train.shape[0], size=(BS))
      xb, yb = Tensor(X_train[samp], requires_grad=False), Tensor(Y_train[samp])

      out = model(xb)
      loss = out.sparse_categorical_crossentropy(yb)
      opt.zero_grad()
      loss.backward()
      opt.step()

      y_preds = out.numpy().argmax(axis=-1)
      acc = (y_preds == yb.numpy()).mean()
      if step == 0 or (step + 1) % 20 == 0:
        print(f"Step {step+1:<3} | Loss: {loss.numpy():.4f} | Train Acc: {acc:.3f}")

  # Evaluate
  acc = 0
  for i in range(0, len(Y_test), BS):
    xb, yb = Tensor(X_test[i:i+BS], requires_grad=False), Tensor(Y_test[i:i+BS])
    out = model(xb)
    preds = out.argmax(axis=-1)
    acc += (preds == yb).sum().numpy()
  acc /= len(Y_test)
  print(f"Test Acc: {acc:.3f}")