import torch
from torch import nn
import torch.nn.functional as F
from VarianceSchedule import VarianceSchedule
from tqdm.auto import tqdm


def extract(a, t, x_shape):
    """
    Retrieve specific elements from a given tensor 'a'.
    't' is a tensor containing the indices to be retrieved from tensor 'a'.
    The output of this function is a tensor containing the elements from tensor 'a' corresponding to each index in tensor 't'.
    :param a:
    :param t:
    :param x_shape:
    :return:
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class DiffusionModel(nn.Module):
    def __init__(self, schedule_name="Linear_beta", timesteps=1000, beta_start=0.0001, beta_end=0.2,
                 denoise_model=None):
        super(DiffusionModel, self).__init__()

        self.denoise_model = denoise_model

        variance_schedule_func = VarianceSchedule(schedule_name=schedule_name, beta_start=beta_start, beta_end=beta_end)

        self.timesteps = timesteps
        self.betas = variance_schedule_func(timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)


        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_recip=torch.sqrt(1.0/self.alphas)
        self.sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0-self.alphas_cumprod)

        self.posterior_variance=self.betas*(1.0-self.alphas_cumprod_prev)/(1.0-self.alphas_cumprod)

    def q_sample(self,x_start,t,noise=None):
        #forward diffusion
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t=extract(self.sqrt_alphas_cumprod,t,x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self,x_t,t,t_index):
        betas_t=extract(self.betas,t,x_t.shape)
        sqrt_one_minus_alphas_cumprod_t=extract(self.sqrt_one_minus_alphas_cumprod,t,x_t.shape)
        sqrt_recip_alphas_t=extract(self.sqrt_alphas_recip,t,x_t.shape)

        mean=sqrt_recip_alphas_t*(x_t-betas_t*self.denoise_model(x_t,t)/sqrt_one_minus_alphas_cumprod_t)
        variance=extract(self.posterior_variance,t,x_t.shape)

        if t_index==0:
            return mean
        else: return mean+variance

    @torch.no_grad()
    def compute_loss(self,x_start,t,loss_type):
        noise=torch.randn_like(x_start)

        forward_noise=self.q_sample(x_start=x_start,t=t,noise=noise)
        model_noise=self.denoise_model(forward_noise,t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, model_noise)
        elif loss_type == 'mse':
            loss = F.mse_loss(noise, model_noise)
        elif loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(noise, model_noise)
        else:
            raise NotImplementedError()

        return loss

    def p_sample_loop(self,shape):
        device=next(self.denoise_model.parameters()).device

        b=shape[0]

        img=torch.randn(shape,device=device)
        imgs=[]

        """
        start from noise
        """

        for i in tqdm(reversed(range(0,self.timesteps)),desc="sampling loop time step",total=self.timesteps):
            img=self.p_sample(img,torch.full((b,),i,device=device,dtype=torch.long),i)
            imgs.append(img.cpi().numpy())
            return imgs

    @torch.no_grad()
    def sample(self,image_size,batch_size=16,channels=3):
        return self.p_sample_loop(shape=(batch_size,channels,image_size,image_size))

    def forward(self, mode, **kwargs):
        if mode == "train":
            if "x_start" not in kwargs or "t" not in kwargs:
                raise ValueError("The diffusion model must be provided with the parameters x_start and t during training!")

            x_start = kwargs["x_start"]
            t = kwargs["t"]
            loss_type = kwargs.get("loss_type", None)
            noise = kwargs.get("noise", None)

            return self.compute_loss(x_start=x_start, t=t, loss_type=loss_type, noise=noise)

        elif mode == "generate":
            if "image_size" and "batch_size" and "channels" in kwargs.keys():
                return self.sample(image_size=kwargs["image_size"],batch_size=kwargs["batch_size"],channels=kwargs["channels"])
            else:
                raise ValueError("During image generation, the diffusion model must be provided with three parameters: image_size, batch_size, and channels.")

        else:
            raise ValueError("only two patterns to be chosen:train/generate")