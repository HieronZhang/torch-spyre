import torch
import torch_spyre  # noqa: F401
from torch.profiler import ProfilerActivity, profile


def main():
    device = torch.device("spyre")

    def add(x, y):
        return x + y

    compiled_add = torch.compile(add)
    x = torch.randn(16, 16, dtype=torch.float16, device=device)
    y = torch.randn(16, 16, dtype=torch.float16, device=device)

    compiled_add(x, y).cpu()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        compiled_add(x, y).cpu()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()

