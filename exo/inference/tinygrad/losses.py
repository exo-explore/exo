from tinygrad import Tensor, dtypes
def length_masked_ce_loss(model, inputs, targets, lengths):
  # Run model on inputs
  logits = model(inputs)
  logits = logits.cast(dtypes.float32)

  # Mask padding tokens
  length_mask = Tensor.arange(inputs.shape[1])[None, :] < lengths[:, None]

  # Calculate the loss
  ce = logits.sparse_categorical_crossentropy(targets) * length_mask
  ntoks = length_mask.sum()
  ce = ce.sum() / ntoks
  return ce

