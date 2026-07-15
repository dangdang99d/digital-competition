"""F1-gate the shippable int8 ONNX: build engine on this GPU, score the 3.5k val slice."""
import sys, time, numpy as np, tensorrt as trt
sys.path.insert(0, '/ws/experiments/compression/quantization')
import script_trt as S

W = '/ws/experiments/compression/quantization/_work'
logger = trt.Logger(trt.Logger.ERROR)

d = np.load(f'{W}/val_tokens.npz')
ids, mask, y = d['val_ids'], d['val_mask'], d['val_y']

b = trt.Builder(logger)
net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
p = trt.OnnxParser(net, logger)
assert p.parse(open(f'{W}/model_int8_ship.onnx', 'rb').read()), 'parse failed'
cfg = b.create_builder_config()
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
cfg.set_flag(trt.BuilderFlag.FP16)
cfg.set_flag(trt.BuilderFlag.INT8)
prof = b.create_optimization_profile()
for n in ('input_ids', 'attention_mask'):
    prof.set_shape(n, (1, 512), (64, 512), (64, 512))
cfg.add_optimization_profile(prof)
t0 = time.time()
blob = b.build_serialized_network(net, cfg)
assert blob is not None
print(f'build {time.time()-t0:.1f}s, engine {blob.nbytes/1e6:.0f}MB', flush=True)
eng = trt.Runtime(logger).deserialize_cuda_engine(blob)
ctx = eng.create_execution_context()

B = 64
di, dm, do = S.cuda_malloc(B*512*8), S.cuda_malloc(B*512*8), S.cuda_malloc(B*14*4)
out = np.empty((B, 14), np.float32)
lo = np.empty((len(ids), 14), np.float32)
t0 = time.time()
for s in range(0, len(ids), B):
    e = min(s+B, len(ids)); bb = e-s
    S.h2d(di, np.ascontiguousarray(ids[s:e].astype(np.int64)))
    S.h2d(dm, np.ascontiguousarray(mask[s:e].astype(np.int64)))
    ctx.set_input_shape('input_ids', (bb, 512)); ctx.set_input_shape('attention_mask', (bb, 512))
    ctx.set_tensor_address('input_ids', di); ctx.set_tensor_address('attention_mask', dm)
    ctx.set_tensor_address('logits', do)
    ctx.execute_async_v3(0); S.CUDART.cudaDeviceSynchronize()
    S.d2h(out[:bb], do); lo[s:e] = out[:bb]
print(f'infer {time.time()-t0:.1f}s', flush=True)

from sklearn.metrics import f1_score
pred = lo.argmax(-1)
print(f'F1GATE F1_val={f1_score(y, pred, average="macro"):.4f} classes={len(np.unique(pred))}/14 finite={np.isfinite(lo).all()}', flush=True)
