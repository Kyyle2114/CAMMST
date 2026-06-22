import math
from typing import List, Union
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    """
    Cosine Annealing with Warm Restarts and Warm Up scheduler.
    
    This scheduler implements a combination of:
    - Linear warm-up for the first T_up iterations.
    - Cosine annealing for the remaining iterations.
    - Restarts capability with optional cycle length multiplier.
    """
    def __init__(self, optimizer: Optimizer, T_0: int, T_mult: int = 1, 
                 eta_max: float = 0.1, T_up: int = 0, gamma: float = 1., 
                 last_epoch: int = -1) -> None:
        """
        Initialize the Cosine Annealing with Warm Restarts scheduler.
        
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_0 (int): Number of iterations for the first restart (can be steps or epochs).
            T_mult (int): Factor to increase T_i after a restart (must be >= 1).
            eta_max (float): Maximum learning rate.
            T_up (int): Number of iterations for linear warm-up (can be steps or epochs).
            gamma (float): Decrease rate of max learning rate by cycle.
            last_epoch (int): The index of last iteration.
            
        Raises:
            ValueError: If T_0 <= 0, T_mult < 1, T_up < 0, or eta_max <= 0.
            
        Example:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            >>> scheduler = CosineAnnealingWarmUpRestarts(
            ...     optimizer, T_0=150, T_mult=1, eta_max=0.1, T_up=10
            ... )
            >>> for _ in range(num_epochs):
            ...     for _ in range(num_steps_per_epoch):
            ...         scheduler.step()
            ...         train(...)
            ...     validate(...)
        """
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError(f"Expected non-negative integer T_up, but got {T_up}")
        if eta_max <= 0:
            raise ValueError(f"Expected positive eta_max, but got {eta_max}")
        if T_up > T_0:
            raise ValueError(f"T_up ({T_up}) must be less than or equal to T_0 ({T_0})")
            
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """
        Calculate current learning rates for all parameter groups.
        
        Returns:
            List[float]: List of learning rates for each parameter group
        """
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            # linear warm-up phase
            return [
                (self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr 
                for base_lr in self.base_lrs
            ]
        else:
            # cosine annealing phase
            return [
                base_lr + (self.eta_max - base_lr) * 
                (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch: Union[int, None] = None) -> None:
        """
        Update learning rates. This should be called after every batch update.
        
        Args:
            epoch (Union[int, None]): Not used, only for compatibility. Set to None.
            
        Raises:
            ValueError: If epoch is negative.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
    def get_last_lr(self) -> List[float]:
        """
        Return last computed learning rates by current scheduler.
        
        Returns:
            List[float]: List of last computed learning rates
        """
        return self.get_lr()
    
    def state_dict(self) -> dict:
        """
        Return the state of the scheduler as a dict.
        
        Returns:
            dict: Scheduler state dictionary
        """
        return {
            'T_0': self.T_0,
            'T_mult': self.T_mult,
            'base_eta_max': self.base_eta_max,
            'eta_max': self.eta_max,
            'T_up': self.T_up,
            'T_i': self.T_i,
            'gamma': self.gamma,
            'cycle': self.cycle,
            'T_cur': self.T_cur,
            'last_epoch': self.last_epoch,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the scheduler's state.
        
        Args:
            state_dict (dict): Scheduler state dictionary
            
        Raises:
            KeyError: If required keys are missing from state_dict
        """
        required_keys = [
            'T_0', 'T_mult', 'base_eta_max', 'eta_max', 'T_up', 
            'T_i', 'gamma', 'cycle', 'T_cur', 'last_epoch', 'base_lrs'
        ]
        
        for key in required_keys:
            if key not in state_dict:
                raise KeyError(f"Missing required key '{key}' in state_dict")
        
        self.__dict__.update(state_dict)