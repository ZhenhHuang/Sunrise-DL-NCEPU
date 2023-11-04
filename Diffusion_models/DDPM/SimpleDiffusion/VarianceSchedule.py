import torch
import math
from torch import nn

# def cosine_beta_schedule(timesteps,s=0.008,**kwargs):
#     steps=timesteps+1
#     x=torch.linspace(0,timesteps,steps)
#     alphas_cumprod=torch.cos(((x/timesteps)+s)/(1+s)*math.pi*0.5)**2
#     alphas_cumprod=alphas_cumprod/alphas_cumprod[0]
#     betas=1-(alphas_cumprod[1:]/alphas_cumprod[:-1])
#     return torch.clip(betas,0.0001,0.9999)

def linear_beta_schedule(timesteps,beta_start=0.0001,beta_end=0.02):
    return torch.linspace(beta_start,beta_end,timesteps)

def quadratic_beta_schedule(timesteps,beta_start=0.0001,beta_end=0.02):
    return torch.linspace(beta_start**0.5,beta_end**0.5,timesteps)**2

def sigmoid_beta_schedule(timesteps,beta_start=0.0001,beta_end=0.02):
    betas=torch.linspace(-6,6,timesteps)
    return torch.sigmoid(betas)*(beta_end-beta_start)+beta_start

class VarianceSchedule(nn.Module):
    def __init__(self,schedule_name,beta_start=None,beta_end=None):
        super(VarianceSchedule, self).__init__()

        self.schedule_name=schedule_name

        # beta  dic
        beta_schedule_dict = {
            'linear_beta_schedule': linear_beta_schedule,
            # 'cosine_beta_schedule': cosine_beta_schedule,
            'quadratic_beta_schedule': quadratic_beta_schedule,
            'sigmoid_beta_schedule': sigmoid_beta_schedule
        }

        if schedule_name in beta_schedule_dict:
            self.selected_schedule=beta_schedule_dict[schedule_name]
        else:
            raise ValueError("schedule_name is nod found")

        self.beta_start=beta_start
        self.beta_end=beta_end

    def forward(self,timesteps):
        return self.selected_schedule(timesteps=timesteps,beta_start=self.beta_start,beta_end=self.beta_end)