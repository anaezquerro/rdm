from __future__ import annotations
import torch 
from torch.nn.functional import conv1d, pad


class TimeSampler:
    def __init__(
        self, 
        device: torch.device,
        n: int = 100000, 
        eps: float = 1e-6
    ):
        self.device = device 
        self.hist = HistogramPDFEstimator(self.sample(n))
        self.eps = eps 
        
    def to(self, device) -> TimeSampler:
        self.device = device 
        self.hist.histogram = self.hist.histogram.to(self.device)
        return self 
    
    def sample(self, n: int) -> torch.Tensor:
        return torch.rand(n, device=self.device)
    
    def __call__(self, n: int) -> torch.Tensor:
        t = self.sample(n)
        probs =  self.hist(t).to(self.device)
        weights = (1 + self.eps) / (probs + self.eps)
        return t, weights
    
    @classmethod
    def from_conf(cls, name: str, **kwargs) -> TimeSampler:
        if name == 'beta':
            return Beta(**kwargs)
        else:
            return cls(**kwargs)
    
class HistogramPDFEstimator:
    """Estimates the density of a set of samples and returns the inverse of the density as weights."""

    def __init__(
        self,
        samples: torch.Tensor,
        num_bins: int = 1000,
        blur_kernel_size: int = 5,
        blur_kernel_sigma: float = 0.2,
        min_weight: float = 1e-2
    ) :
        self.num_bins = num_bins 
        self.blur_kernel_size = blur_kernel_size 
        self.blur_kernel_sigma = blur_kernel_sigma 
        self.histogram = self.get_smooth_density_histogram(samples).clamp(min_weight)

    def get_gaussian_1d_kernel(self) -> torch.Tensor:
        if self.blur_kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        # Create a range of values centered at 0
        center = self.blur_kernel_size // 2
        x = torch.arange(-center, center + 1, dtype=torch.float32)

        # Compute the Gaussian function
        kernel = torch.exp(-0.5 * (x / self.blur_kernel_sigma) ** 2)

        # Normalize the kernel to ensure sum equals 1
        kernel /= kernel.sum()
        return kernel

    def get_smooth_density_histogram(self, vals: torch.Tensor) -> torch.Tensor:
        assert vals.min() >= 0, "Timesteps must be nonnegative"
        assert vals.max() <= 1, "Timesteps must be less or equal than 1"
        histogram_torch = torch.histc(vals, self.num_bins, min=0, max=1).to(vals.device)
        kernel = self.get_gaussian_1d_kernel().to(vals.device)

        # Reflective padding to avoid edge effects in convolution
        padded_hist = pad(
            histogram_torch.unsqueeze(0).unsqueeze(0),
            (self.blur_kernel_size // 2, self.blur_kernel_size // 2),
            mode="reflect",
        )
        histogram_torch_conv = conv1d(
            padded_hist, kernel.unsqueeze(0).unsqueeze(0)
        ).to(vals.device)

        # remove unnecessary dimensions and normalize to pdf
        return histogram_torch_conv.squeeze() / histogram_torch_conv.mean()

    def __call__(self, t: torch.Tensor):
        bin_ids = (t * self.num_bins).long()
        bin_ids.clamp_(0, self.num_bins - 1)
        return self.histogram[bin_ids]
    


class Beta(TimeSampler):
    def __init__(self, alpha: float, beta: float, **kwargs):
        self.beta = torch.distributions.beta.Beta(alpha, beta)
        super(Beta, self).__init__(**kwargs)
        
    def sample(self, n: int) -> torch.Tensor:
        return self.beta.sample((n,)).to(self.device)
