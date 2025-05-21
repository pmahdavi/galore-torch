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
        loss = None
        if closure is not None:
            with torch.enable_grad(): # Ensure grads are enabled for closure
                loss = closure()

        # Handle fused groups first
        # _apply_fused_update will iterate through self.param_groups and
        # only process groups that have group.get("fused", False) == True.
        # For these groups, p.grad is cleared by the hook _fused_accumulate_grad_hook
        # and the update uses state["projected_grad_accum"].
        any_fused_groups = any(group.get("fused", False) for group in self.param_groups)
        if any_fused_groups:
            self._apply_fused_update()

        # Now, handle non-fused groups, or all groups if no group was fused.
        # This logic relies on p.grad being available.
        # If any_fused_groups is true, this loop will skip the already processed fused groups.
        for group in self.param_groups:
            if group.get("fused", False): # If fused, it was handled by _apply_fused_update
                continue

            # Standard non-fused update logic for this non-fused group (or any group if no fusion at all)
            for p in group["params"]:
                if p.grad is None:
                    # For non-fused parameters, grad should exist if computed.
                    # If it's None, it means no gradient for this param in this step.
                    continue

                grad = p.grad # Original gradient
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                
                # Initialize step for this parameter if it's the first time
                if "step" not in state:
                    state["step"] = 0
                
                # GaLore Projection: Apply if "rank" is in group.
                # This handles cases where GaLore is used for a group but not in fused mode.
                # For truly non-GaLore groups (like embeddings, lm_head from llamafactory),
                # "rank" won't be present, so this block is skipped.
                if "rank" in group:
                    if "projector" not in state:
                        # Default dim for projector if not specified in group
                        if 'dim' not in group:
                            group['dim'] = 2 

                        if group['dim'] <= 2:
                            state["projector"] = GaLoreProjector(
                                group["rank"], 
                                update_proj_gap=group["update_proj_gap"], 
                                scale=group["scale"], 
                                proj_type=group["proj_type"]
                            )
                        else:
                            state["projector"] = GaLoreProjectorTensor(
                                group["rank"], 
                                update_proj_gap=group["update_proj_gap"], 
                                scale=group["scale"], 
                                proj_type=group["proj_type"]
                            )
                    # grad variable is updated to the projected gradient
                    grad = state["projector"].project(grad, state["step"]) 

                # Adam State initialization (using potentially projected grad for shapes)
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1 # Increment step count for this parameter

                # AdamW update computation
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                norm_grad = exp_avg / denom # This is the normalized gradient update
                
                # GaLore Projection Back: Apply if "rank" is in group (matching the projection step)
                if "rank" in group:
                    norm_grad = state["projector"].project_back(norm_grad)
                
                # Apply parameter update
                p.add_(norm_grad, alpha=-step_size)

                # Weight decay application
                if group["weight_decay"] > 0.0:
                    # AdamW's decoupled weight decay
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss

    def _fused_accumulate_grad_hook(self, param: torch.Tensor):
        """
        Called per-parameter per-backward:
        - Project gradient
        - Accumulate projected gradient
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
            
        if param.grad is None: # Should not happen if hook is called, but good practice
            return

        raw_grad = param.grad
        
        # Apply projection if needed
        if "rank" in group:
            projected_grad = state["projector"].project(raw_grad, state.get("step", 0))
        else: # Should not happen for fused GaLore, but as a fallback
            projected_grad = raw_grad.clone() 
            
        # Accumulate projected gradient
        # state["step"] is the current completed optimizer step count.
        # The hook is called for micro-batches of the *next* optimizer step.
        # We use "optimizer_step_for_accumulation" to track which optimizer step these grads belong to.
        
        current_optimizer_step_count = state.get("step", 0)

        if state.get("optimizer_step_for_accumulation", -1) != current_optimizer_step_count:
            # This is the first micro-batch for the upcoming optimizer step (current_optimizer_step_count + 1)
            # Or, the optimizer step has advanced, so we reset accumulation.
            state["projected_grad_accum"] = projected_grad.clone()
            state["optimizer_step_for_accumulation"] = current_optimizer_step_count
        else:
            # Subsequent micro-batch for the same upcoming optimizer step
            if "projected_grad_accum" in state:
                state["projected_grad_accum"].add_(projected_grad)
            else: # Should have been initialized in the if block
                state["projected_grad_accum"] = projected_grad.clone()
        
        # clear raw grad
        param.grad = None

    def _apply_fused_update(self):
        """
        Called on .step() when fused=True:
        - Uses accumulated projected gradient to update moments.
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
                
                # Retrieve accumulated projected gradient
                accumulated_projected_grad = state.pop("projected_grad_accum", None)

                # If no accumulated grad (e.g. param not part of GaLore group with "rank", or no grads received)
                # and no existing moments, skip. If moments exist, it implies it was updated in a previous step,
                # and this step it received no grad. Adam updates will handle this (moments decay).
                if accumulated_projected_grad is None and "exp_avg" not in state :
                    continue

                # increment step count first, as it's used for bias correction
                # and projector's internal step tracking if it relies on optimizer step
                step = state.get("step", 0) + 1
                state["step"] = step
                
                # Initialize moment buffers if they don't exist
                # This handles the case where a parameter might not have received a gradient
                # in the very first step but does now, or if using accumulated_projected_grad
                # for the first time.
                if "exp_avg" not in state:
                    # If accumulated_projected_grad is None here, but moments need init,
                    # it implies an issue or a parameter that was expected to have grads but didn't.
                    # For safety, initialize with zeros of param shape if grad is missing.
                    # However, GaLore params should always have "rank" and thus projector.
                    # The grad for init should ideally be from accumulated_projected_grad.
                    if accumulated_projected_grad is not None:
                        state["exp_avg"]    = torch.zeros_like(accumulated_projected_grad)
                        state["exp_avg_sq"] = torch.zeros_like(accumulated_projected_grad)
                    elif "rank" in group: # GaLore parameter that got no grad, init with zeros based on param shape
                        # This case is tricky. If projector needs a specific shape for its grads,
                        # this might be problematic. Assuming projector outputs gradients of same shape as param for now.
                        # This usually means param.shape if projection is done and then undone, or projected shape.
                        # Since moments are on projected grads, this needs careful thought.
                        # The projector.project gives the shape. Let's assume it's param.shape for Adam moments.
                        # This part of logic matches original code: moments are like the grad.
                        # If accumulated_projected_grad is None, but it's a GaLore param,
                        # it means it got no grad through all accumulation steps.
                        # So, its contribution to moments is zero.
                        state["exp_avg"]    = torch.zeros_like(param.data) # Defaulting to param.data shape
                        state["exp_avg_sq"] = torch.zeros_like(param.data)
                    else: # Non-GaLore param that somehow ended up here and needs init (should not happen for fused GaLore)
                        state["exp_avg"]    = torch.zeros_like(param.data)
                        state["exp_avg_sq"] = torch.zeros_like(param.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                if accumulated_projected_grad is not None:
                    # Update moments using the accumulated projected gradient
                    exp_avg.mul_(beta1).add_(accumulated_projected_grad, alpha=(1 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(accumulated_projected_grad, accumulated_projected_grad, value=(1 - beta2))
                else:
                    # If no accumulated gradient, moments just decay (handled by mul_(beta) if done unconditionally)
                    # For Adam, if grad is zero, exp_avg decays, exp_avg_sq decays.
                    exp_avg.mul_(beta1)
                    exp_avg_sq.mul_(beta2)

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
