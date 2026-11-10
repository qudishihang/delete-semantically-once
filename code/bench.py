"""Benchmark CPU throughput for candidate experimental platforms."""
import time, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("NT", "12")))

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA, exist_ok=True)

print("threads", torch.get_num_threads())

t0 = time.time()
import torchvision
from torchvision import datasets, transforms
tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
train = datasets.CIFAR10(DATA, train=True, download=True, transform=tr)
test = datasets.CIFAR10(DATA, train=False, download=True, transform=tr)
print(f"download+load {time.time()-t0:.1f}s  n_train={len(train)} n_test={len(test)}")

loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, num_workers=0)


class SmallCNN(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64, nc)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)     # 16
        x = F.max_pool2d(F.relu(self.c2(x)), 2)     # 8
        x = F.max_pool2d(F.relu(self.c3(x)), 2)     # 4
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def bench_model(model, name, n_batches=8):
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    t = time.time()
    it = iter(loader)
    for _ in range(n_batches):
        x, y = next(it)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    dt = time.time() - t
    per_b = dt / n_batches
    print(f"{name}: {per_b*1000:.0f} ms/batch -> {per_b*len(loader)/60:.2f} min/epoch (50k)")
    return per_b


print("--- model throughput ---")
bench_model(SmallCNN(), "SmallCNN")


class ResNetCifar(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        import torchvision.models as M
        m = M.resnet18(num_classes=nc)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        self.m = m

    def forward(self, x):
        return self.m(x)


bench_model(ResNetCifar(), "ResNet18-CIFAR")

print("--- inference only (feature extraction) ---")
m = ResNetCifar().eval()
t = time.time()
with torch.no_grad():
    for i, (x, y) in enumerate(loader):
        m(x)
        if i >= 8:
            break
print(f"  fwd-only {(time.time()-t)/9*1000:.0f} ms/batch")

print("--- linear probe on frozen features ---")
import torchvision.models as M
enc = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
enc.fc = nn.Identity()
for p in enc.parameters():
    p.requires_grad_(False)
enc.eval()


def extract(ds, bs=128):
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)
    feats, ys = [], []
    t = time.time()
    with torch.no_grad():
        for x, y in dl:
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
            feats.append(enc(x))
            ys.append(y)
    print(f"  extract {len(ds)} imgs in {time.time()-t:.1f}s")
    return torch.cat(feats), torch.cat(ys)


Xtr, ytr = extract(train)
Xte, yte = extract(test)
print("  feat shape", Xtr.shape)

# time a linear-head epoch
head = nn.Linear(512, 10)
opt = torch.optim.SGD(head.parameters(), lr=1.0, momentum=0.9)
t = time.time()
for _ in range(10):
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 256):
        idx = perm[i:i + 256]
        opt.zero_grad()
        F.cross_entropy(head(Xtr[idx]), ytr[idx]).backward()
        opt.step()
print(f"  linear epoch {time.time()-t:.2f}s (10 epochs) -> {(time.time()-t)/10:.3f}s/epoch")
with torch.no_grad():
    acc = (head(Xte).argmax(1) == yte).float().mean().item()
print(f"  linear-probe test acc {acc:.4f}")
