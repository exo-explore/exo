import mlx.core as mx
import mlx.nn as nn
def length_masked_ce_loss(model, inputs, targets, lengths):
  # Run model on inputs
  logits = model(inputs).astype(mx.float32)

  # Mask padding tokens
  length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

  # Calculate the loss
  ce = nn.losses.cross_entropy(logits, targets) * length_mask
  loss = ce.sum() / length_mask.sum()
  return loss

#Naive intermediate layer loss, where we replace the targets with gradients and just multiply the output by the gradients to derive the loss. This is naive and may warrant some further iteration, but will do the job for now
def back_gradient_loss(model, inputs, gradients, lengths):
  out = model(inputs).astype(mx.float32)
  length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

  approximation = (out.sum(axis=-1) * length_mask * gradients)
  loss = approximation.sum() / length_mask.sum()
  return loss

loss_fns = {
  "back_gradient": back_gradient_loss,
  "length_masked_ce": length_masked_ce_loss,
}
