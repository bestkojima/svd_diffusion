from config import model_config

from model import Unet

c=model_config()
m=Unet(
    dim=c.dim,
    dim_mults=c.dim_mults,
    channels=c.channels,
)
print(Unet)