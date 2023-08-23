#!/usr/bin/env python3
import pathlib

FILES = ["tensor.py", "mlops.py", "helpers.py", "nn/optim.py", "ops.py", "runtime/lib.py", "runtime/ops_cpu.py", "shape/shapetracker.py", "shape/symbolic.py"]
#FILES += ["lazy.py"]
src = pathlib.Path("../tinygrad/tinygrad")
dest = pathlib.Path("teenygrad")

for f in FILES:
  rd = open(src/f).read()
  rd = rd.split("# --teenygrad--")[0]
  rd = rd.replace("from tinygrad.", "from teenygrad.")
  rd = rd.replace("import tinygrad.", "import teenygrad.")
  rd = rd.replace("tinygrad.runtime.", "teenygrad.runtime.")
  with open(dest/f, "w") as f:
    f.write(rd)
