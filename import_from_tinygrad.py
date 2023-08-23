#!/usr/bin/env python3
import pathlib

FILES = ["tensor.py", "mlops.py", "nn/optim.py"]
src = pathlib.Path("../tinygrad/tinygrad")
dest = pathlib.Path("teenygrad")

for f in FILES:
  rd = open(src/f).read()
  rd = rd.replace("from tinygrad.", "from teenygrad.")
  rd = rd.replace("import tinygrad.", "import teenygrad.")
  with open(dest/f, "w") as f:
    f.write(rd)
