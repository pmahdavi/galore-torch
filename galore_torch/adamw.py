# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version

from .galore_projector import GaLoreProjector
from .galore_projector_tensor import GaLoreProjectorTensor


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        
        # Create param-to-group mapping for hooks
        self._param_to_group = {}
        for group in self.param_groups:
            for p in group["params"]:
                self._param_to_group[id(p)] = group
        
        # register fused-backward hooks if requested *within the group*
        for group in self.param_groups:
            if group.get("fused", False):
                for p in group["params"]:
                    p.register_post_accumulate_grad_hook(
                        self._fused_accumulate_grad_hook
                    )

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        # fused feature: delegate to fused-update if enabled
        if any(group.get("fused", False) for group in self.param_groups):
            return self._apply_fused_update()

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                
                if 'dim' not in group:
                    group['dim'] = 2
                    
                # GaLore Projection
                if "rank" in group:
                    if "projector" not in state:
                        if group['dim'] <=2:
                            state["projector"] = GaLoreProjector(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
                        else:
                            state["projector"] = GaLoreProjectorTensor(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
                    grad = state["projector"].project(grad, state["step"])

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom
                
                # GaLore Projection Back
                if "rank" in group:
                    norm_grad = state["projector"].project_back(norm_grad)
                
                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss

    def _fused_accumulate_grad_hook(self, param: torch.Tensor):
        """
        Called per-parameter per-backward:
        - Project/update GaLore moment buffers
        - Clear raw grad to free memory
        """
        state = self.state[param]
        
        # Get parameter group using stored mapping
        group = self._param_to_group[id(param)]
        
        # Set default dim if not present
        if 'dim' not in group:
            group['dim'] = 2
            
        # lazy init projector
        if "rank" in group and "projector" not in state:
            ctor = (GaLoreProjector if group["dim"] <= 2 else GaLoreProjectorTensor)
            state["projector"] = ctor(
                group["rank"],
                update_proj_gap=group["update_proj_gap"],
                scale=group["scale"],
                proj_type=group["proj_type"],
            )
            
        grad = param.grad
        # apply projection if needed
        if "rank" in group:
            grad = state["projector"].project(grad, state.get("step", 0))
            
        # init moment buffers
        if "exp_avg" not in state:
            state["exp_avg"]    = torch.zeros_like(grad)
            state["exp_avg_sq"] = torch.zeros_like(grad)
            
        beta1, beta2 = group["betas"]
        
        # check "last_iter" in state, and if does not exist set it to -1
        if "last_iter" not in state:
            state["last_iter"] = -1

        # Initialize step if it doesn't exist
        if "step" not in state:
             state["step"] = 0

        # update moments conditional on being in first iteration of gradient accumulation.
        # we check that by comparing "step" key in state to last_iter
        if state["step"] != state["last_iter"]:
            state["exp_avg"].mul_(beta1).add_(grad, alpha=(1 - beta1))
            state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
            state["last_iter"] = state["step"]
        else:
            # if not in first iteration, just update moments
            state["exp_avg"].add_(grad, alpha=(1 - beta1))
            state["exp_avg_sq"].addcmul_(grad, grad, value=(1 - beta2))
        
        # clear raw grad
        param.grad = None

    def _apply_fused_update(self):
        """
        Called on .step() when fused=True:
        - Performs bias-correct, back-project, weight add, weight-decay
        """
        for group in self.param_groups:
            if not group.get("fused", False):
                continue
                
            lr, eps, wd = (group[k] for k in ("lr","eps","weight_decay"))
            correct = group["correct_bias"]
            beta1, beta2 = group["betas"]
            
            for param in group["params"]:
                state = self.state[param]
                
                # Skip parameters that didn't receive gradients (e.g., if they were
                # in a different branch that wasn't executed)
                if "exp_avg" not in state:
                    continue
                    
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                
                # increment step count
                step = state.get("step", 0) + 1
                state["step"] = step
                
                # bias correction
                if correct:
                    bias1 = 1 - beta1 ** step
                    bias2 = 1 - beta2 ** step
                    step_size = lr * (math.sqrt(bias2) / bias1)
                else:
                    step_size = lr
                    
                denom = exp_avg_sq.sqrt().add_(eps)
                norm_grad = exp_avg.div(denom)
                
                if "rank" in group:
                    norm_grad = state["projector"].project_back(norm_grad)
                    
                param.add_(norm_grad, alpha=-step_size)
                
                if wd > 0:
                    param.add_(param, alpha=-lr * wd)
                    
        return None
