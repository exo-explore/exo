import os, atexit, functools, contextlib
from collections import defaultdict
from typing import List, Any, DefaultDict, Union
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MetaOps, BufferOps, TernaryOps, LazyOp
from tinygrad.device import Device
from tinygrad.helpers import GRAPHPATH, DEBUG, GlobalCounters, getenv
from tinygrad.codegen.uops import UOps, UOp
from tinygrad.codegen.uopgraph import UPat
from tinygrad.shape.symbolic import NumNode
from tinygrad.lazy import LazyBuffer

with contextlib.suppress(ImportError): import networkx as nx

# **** debugging and graphing ****

if DEBUG >= 2:
  def print_globalcounters():
    if GlobalCounters.time_sum_s == 0: return
    print(f"avg: {GlobalCounters.global_ops*1e-9/GlobalCounters.time_sum_s:8.2f} GFLOPS {GlobalCounters.global_mem*1e-9/GlobalCounters.time_sum_s:8.2f} GB/s",  # noqa: E501
          f"{' '*10}total: {GlobalCounters.kernel_count:5d} kernels {GlobalCounters.global_ops*1e-9:8.2f} GOPS {GlobalCounters.global_mem*1e-9:8.2f} GB {GlobalCounters.time_sum_s*1e3:8.2f} ms")  # noqa: E501
  atexit.register(print_globalcounters)

def save_graph(G, fn, opt=""):
  print("saving", G, f"to {fn}.svg")
  nx.drawing.nx_pydot.write_dot(G, f'{fn}.dot')
  os.system(f'dot {opt} -Tsvg {fn}.dot -o {fn}.svg')

G:Any = None
def init_graph():
  global G
  if G is not None: return
  G = nx.DiGraph()
  atexit.register(functools.partial(save_graph, G, GRAPHPATH)) # -Gnslimit=100 can make it finish, but you won't like results

counts: DefaultDict[type, int] = defaultdict(int)
def nm(x):
  if not hasattr(x, 'node_id'):
    setattr(x, 'node_id', counts[type(x)])
    counts[type(x)] += 1
  return x.node_id

def realized_lazybuffer(lb:'LazyBuffer', num):
  init_graph()
  G.nodes[nm(lb)]['style'] = '"filled,bold"'
  G.nodes[nm(lb)]['fillcolor'] = G.nodes[nm(lb)]['fillcolor'][:-2]
  G.nodes[nm(lb)]['label'] = '"' + G.nodes[nm(lb)]["label"].replace('"', '') + f'\nK:{num}"'

top_colors = {MetaOps: '#FFFFa0', UnaryOps: "#c0c0c0", ReduceOps: "#FFA0A0", BinaryOps: "#c0c0c0",
              TernaryOps: "#c0c0c0", BufferOps: '#a0a0ff'}
def log_lazybuffer(lb:'LazyBuffer', scheduled=False):
  init_graph()
  if lb.base.realized is None and lb.base.op is MetaOps.CONST: return
  if lb.base != lb:
    offset = lb.st.expr_idxs([NumNode(0)] * len(lb.st.shape))[0]
    label = f"{lb.st.shape}\n{lb.st.real_strides()}" + (f"\n{offset}" if offset != 0 else "")
    G.add_node(nm(lb), style='"filled,dashed"', fillcolor="#80ff8080", color="black", label=label)
    G.add_edge(nm(lb.base), nm(lb), color='#00000060')
    lb = lb.base
  if lb.realized is None:
    label_append = []
    for idx,x in enumerate(lb.srcs):
      if nm(x) not in G.nodes: log_lazybuffer(x)
      if x.base.realized is None and x.base.op is MetaOps.CONST:
        label_append.append(f"\nCONST{idx} {x.base.arg:g}")
      else:
        G.add_edge(nm(x), nm(lb), color='#a0a0a0')
    label = '"' + \
      (str(set(x.shape for x in lb.srcs))+"\n"+str(lb.shape) if lb.op in ReduceOps else str(lb.shape)) + \
      (f"\n{lb.dtype.name}" if lb.dtype.name != "float" else "")+f"\n{lb.op}"+(f"\n{lb.arg}" if lb.op in {MetaOps.CONST, UnaryOps.CAST} else "") + \
      (f"\n{lb.device}" if lb.device != Device.DEFAULT else "") + ''.join(label_append) + f'\n{lb.metadata}"'
    G.add_node(nm(lb), style='"filled,dashed"', fillcolor=[v for k,v in top_colors.items() if lb.op in k][0] + "80", color="black", label=label)
    if scheduled: G.nodes[nm(lb)]['shape'] = 'box'
  else:
    if nm(lb) not in G.nodes:
      # realized but unseen?
      G.add_node(nm(lb), label=f'"{str(lb.base.realized)[5:-1].replace(" ", chr(10))}\nb:{nm(lb.realized)}"', style='filled', fillcolor="#f0c08080")

def _tree(dag:Union[LazyOp, UOp, UPat], cycles, cnt):
  cnt[0] += 1
  src = dag.src if isinstance(dag.src, (list, tuple)) else [] if dag.src is None else [dag.src]
  if len(src) == 0: return [f"━━ {dag.op} {dag.arg}"]
  if (lid := id(dag)) in cycles and cycles[lid][1] > (tcnt := getenv("TREE_CYCLE_CNT", 5)) and tcnt >= 0:
    return [f"━⬆︎ goto {cycles[id(dag)][0]}: {dag.op}"]
  cycles[lid] = (cnt[0], 1 if lid not in cycles else cycles[lid][1]+1)
  lines = [f"━┳ {dag.op} {dag.arg}"]
  childs = [_tree(c, cycles, cnt) for c in src]
  for c in childs[:-1]: lines += [f" ┣{c[0]}"] + [f" ┃{l}" for l in c[1:]]
  return lines + [" ┗"+childs[-1][0]] + ["  "+l for l in childs[-1][1:]]

def print_tree(dag:Union[LazyOp, UOp, UPat]): print("\n".join([f"{str(i).rjust(3)} {s}" for i,s in enumerate(_tree(dag, {}, [-1]))]))

def graph_uops(uops:List[UOp]):
  colors = {UOps.ALU: "#ffffc0", UOps.LOAD: "#ffc0c0", UOps.STORE: "#c0ffc0", UOps.SPECIAL: "#c0c0ff", UOps.CONST: "#e0e0e0",
            UOps.DEFINE_GLOBAL: "#ffe0b0", UOps.DEFINE_LOCAL: "#ffe0d0", UOps.DEFINE_ACC: "#f0ffe0", UOps.REDUCE: "#C4A484",
            UOps.RANGE: "#c8a0e0", UOps.PHI: "#e0ffc0", UOps.BARRIER: "#ff8080", UOps.IF: "#c8b0c0"}
  G = nx.DiGraph()
  for u in uops:
    if u.op in {UOps.ENDRANGE, UOps.ENDIF}: continue
    G.add_node(uops.index(u), label=f"{str(u.op)[5:]}{(' '+str(u.arg).replace(':', '')) if u.arg is not None else ''}\n{str(u.dtype)}", style="filled", fillcolor=colors.get(u.op, "#ffffff"))  # noqa: E501
    for v in u.src: G.add_edge(uops.index(v), uops.index(u))
  save_graph(G, f'{GRAPHPATH}.uops', '-Grankdir=LR')
