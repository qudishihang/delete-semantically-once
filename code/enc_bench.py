import time, os, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(12)
import torchvision.models as M
m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
m.fc = nn.Identity(); m.eval()
for p in m.parameters(): p.requires_grad_(False)
mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
for size in (224, 128, 96, 64, 48):
    x = torch.rand(64,3,size,size)
    with torch.no_grad():
        m((x-mean)/std)  # warm
        t=time.time()
        for _ in range(3):
            m((x-mean)/std)
        dt=(time.time()-t)/3
    print(f"size {size}: {dt/64*1000:.2f} ms/img -> 50k in {dt/64*50000/60:.1f} min", flush=True)
