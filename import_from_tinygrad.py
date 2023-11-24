#!/usr/bin/env python3
import pathlib

FILES = ["tensor.py", "mlops.py", "nn/optim.py", "../test/test_ops.py", "../test/test_dtype.py", "../test/test_optim.py"]
src = pathlib.Path("../tinygrad/tinygrad")
dest = pathlib.Path("teenygrad")

for f in FILES:
  print("importing", f)
  rd = open(src/f).read()
  rd = rd.replace("from tinygrad.", "from teenygrad.")
  rd = rd.replace("import tinygrad.", "import teenygrad.")
  (dest/f).parent.mkdir(parents=True, exist_ok=True)
  with open(dest/f, "w") as f:
    f.write(rd)
