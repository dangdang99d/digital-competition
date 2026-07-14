import sys, time, numpy as np, tensorrt as trt
sys.path.insert(0, '/ws/experiments/compression/quantization')
import script_trt as S
W = '/ws/experiments/compression/quantization/_work'
logger = trt.Logger(trt.Logger.ERROR)
d = np.load(f'{W}/tokens_512.npz'); ii, am = d['input_ids'], d['attention_mask']
blob = S.build_engine(f'{W}/model_fp16.onnx', 64, logger)
eng = trt.Runtime(logger).deserialize_cuda_engine(blob); ctx = eng.create_execution_context()
B = 64; di = S.cuda_malloc(B*512*8); dm = S.cuda_malloc(B*512*8); do = S.cuda_malloc(B*14*4)
out = np.empty((B, 14), np.float32); lo = np.empty((len(ii), 14), np.float32)
t0 = time.time()
for s in range(0, len(ii), B):
    e = min(s+B, len(ii)); b = e-s
    S.h2d(di, np.ascontiguousarray(ii[s:e].astype(np.int64))); S.h2d(dm, np.ascontiguousarray(am[s:e].astype(np.int64)))
    ctx.set_input_shape('input_ids', (b, 512)); ctx.set_input_shape('attention_mask', (b, 512))
    ctx.set_tensor_address('input_ids', di); ctx.set_tensor_address('attention_mask', dm); ctx.set_tensor_address('logits', do)
    ctx.execute_async_v3(0); S.CUDART.cudaDeviceSynchronize(); S.d2h(out[:b], do); lo[s:e] = out[:b]
print(f'infer {time.time()-t0:.1f}s')
np.save(f'{W}/trt_fp16onnx_logits.npy', lo.astype(np.float16))
print('SAVED')
