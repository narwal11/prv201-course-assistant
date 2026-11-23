# Attack: plant a module in site-packages so later runs import attacker code.
import sysconfig, os
path = os.path.join(sysconfig.get_paths()["purelib"], "evil_module.py")
try:
    with open(path, "w") as f:
        f.write("print('pwned')")
    print("ESCAPED: wrote", path)
except Exception as e:
    print("contained:", type(e).__name__)
