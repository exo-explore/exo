from typing import Union, Optional
from tinygrad import Tensor, Variable

class IdentityBlock:
  def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor, mask: Optional[Tensor]):
    return x