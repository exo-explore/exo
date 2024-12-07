from tinygrad import Tensor, dtypes
import numpy as np
def length_masked_ce_loss(model, inputs, targets, lengths):
  # Run model on inputs
  logits = model(inputs).cast(dtypes.float32).contiguous()

  # Mask padding tokens
  length_mask = Tensor(np.arange(inputs.shape[1])[None, :] < lengths[:, None], requires_grad=False)

  # Calculate the loss
  ce = logits.sparse_categorical_crossentropy(Tensor(targets, requires_grad=False)).mul(length_mask)
  loss = ce.sum() / length_mask.sum()
  return loss

