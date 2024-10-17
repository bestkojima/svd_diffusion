
from dataclasses import dataclass, field
@dataclass
class model_config():
    dim = 64
    dim_mults = (1, 2, 4, 8)
    channels=1

@dataclass
class mnist_config():
    img_size=(28,28)
    channels=1
    numsteps=1000



if __name__ == "__main__":
    c=model_config()
    print(type(c.dim))
    print(type(c.dim_mults))
    print(type(c.channels))