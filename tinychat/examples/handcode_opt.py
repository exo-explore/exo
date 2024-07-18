from typing import List
from extra.models.resnet import ResNet50
from examples.mlperf.helpers import get_mlperf_bert_model
from tinygrad import Tensor, Device, dtypes, nn
from tinygrad.codegen.kernel import Kernel
from tinygrad.device import Compiled
from tinygrad.engine.graph import print_tree
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.search import time_linearizer, beam_search, bufs_from_lin
from tinygrad.helpers import DEBUG, ansilen, getenv
from tinygrad.ops import MetaOps, get_lazyop_info
from tinygrad.shape.symbolic import sym_infer


def get_sched_resnet():
  mdl = ResNet50()
  optim = (nn.optim.LARS if getenv("LARS") else nn.optim.SGD)(nn.state.get_parameters(mdl))
  BS = getenv("BS", 64)

  # run model twice to get only what changes, these are the kernels of the model
  seen = set()
  for _ in range(2):
    out = mdl(Tensor.empty(BS, 3, 224, 224))
    targets = [out.lazydata]
    if getenv("BACKWARD"):
      optim.zero_grad()
      out.sparse_categorical_crossentropy(Tensor.empty(BS, dtype=dtypes.int)).backward()
      targets += [x.lazydata for x in optim.schedule_step()]
    sched = create_schedule(targets, seen)
    print(f"schedule length {len(sched)}")
  return sched

def get_sched_bert():
  mdl = get_mlperf_bert_model()
  optim = nn.optim.LAMB(nn.state.get_parameters(mdl))

  # fake data
  BS = getenv("BS", 2)
  input_ids = Tensor.empty((BS, 512), dtype=dtypes.float32)
  segment_ids = Tensor.empty((BS, 512), dtype=dtypes.float32)
  attention_mask = Tensor.empty((BS, 512), dtype=dtypes.default_float)
  masked_positions = Tensor.empty((BS, 76), dtype=dtypes.float32)
  masked_lm_ids = Tensor.empty((BS, 76), dtype=dtypes.float32)
  masked_lm_weights = Tensor.empty((BS, 76), dtype=dtypes.float32)
  next_sentence_labels = Tensor.empty((BS, 1), dtype=dtypes.float32)

  # run model twice to get only what changes, these are the kernels of the model
  seen = set()
  for _ in range(2):
    lm_logits, seq_relationship_logits = mdl(input_ids, attention_mask, masked_positions, segment_ids)
    targets = [lm_logits.lazydata, seq_relationship_logits.lazydata]
    if getenv("BACKWARD"):
      optim.zero_grad()
      loss = mdl.loss(lm_logits, seq_relationship_logits, masked_lm_ids, masked_lm_weights, next_sentence_labels)
      # ignore grad norm and loss scaler for now
      loss.backward()
      targets += [x.lazydata for x in optim.schedule_step()]
    sched = create_schedule(targets, seen)
    print(f"schedule length {len(sched)}")
  return sched

if __name__ == "__main__":
  if getenv("HALF", 1):
    dtypes.default_float = dtypes.half

  # the device we are optimizing for
  device: Compiled = Device[Device.DEFAULT]
  if getenv("BACKWARD"): Tensor.training = True
  print(f"optimizing for {Device.DEFAULT}")

  sched = globals()[f"get_sched_{getenv('MODEL', 'resnet')}"]()
  sched = [x for x in sched if x.ast.op is MetaOps.KERNEL]

  # focus on one kernel
  if getenv("KERNEL", -1) >= 0: sched = sched[getenv("KERNEL", -1):getenv("KERNEL", -1)+1]

  # work with the schedule
  total_tm = 0
  running_gflops = 0
  usage = {}
  for i,si in enumerate(sched):
    ops = get_lazyop_info(si.ast.src[0]).flops

    if DEBUG >= 2:
      print_tree(si.ast)

    rawbufs = bufs_from_lin(Kernel(si.ast))

    # "linearize" the op into uops in different ways
    lins:List[Kernel] = []

    # always try hand coded opt
    lin = Kernel(si.ast, opts=device.renderer)
    lin.hand_coded_optimizations()
    lins.append(lin)

    # maybe try tensor cores
    lin = Kernel(si.ast, opts=device.renderer)
    if lin.apply_tensor_cores():
      lins.append(lin)

    # try a beam search
    if beam:=getenv("BEAM"):
      lin = Kernel(si.ast, opts=device.renderer)
      lin = beam_search(lin, rawbufs, beam, bool(getenv("BEAM_ESTIMATE", 1)))
      lins.append(lin)

    # benchmark the programs
    choices = []
    for lin in lins:
      tm = time_linearizer(lin, rawbufs, allow_test_size=False, cnt=10)
      gflops = sym_infer(ops, {k:k.min for k in lin.ast.vars()})*1e-9/tm
      choices.append((tm, gflops, lin.linearize()))

      # print all kernels
      if DEBUG >= 1: print(f"                 kernel {i:2d} {lin.name+' '*(37-ansilen(lin.name))} {str(lin.global_size):18s} {str(lin.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS")
    tm, gflops, lin = sorted(choices, key=lambda x: x[0])[0]
    total_tm += tm
    running_gflops += gflops * tm
    if (key := str([str(m) for m in si.metadata] if si.metadata is not None else None)) not in usage: usage[key] = (0, 0)
    usage[key] = (usage[key][0] + tm, usage[key][1] + 1)
    print(f"*** {total_tm*1000:7.2f} ms : kernel {i:2d} {lin.name+' '*(37-ansilen(lin.name))} {str(lin.global_size):18s} {str(lin.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS {[str(m) for m in si.metadata] if si.metadata is not None else ''}")
  print(f"******* total {total_tm*1000:.2f} ms, {running_gflops/total_tm:6.0f} GFLOPS")
  print("usage:")
  for k in sorted(usage, key=lambda x: -usage[x][0])[:10]:
    print(f"{usage[k][0]*1000:.2f} ms: {k} ({usage[k][1]} times)")
