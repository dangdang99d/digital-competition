# Quantize the PRUNED fp32 ONNX -> shippable pruned int8 Q/DQ (same recipe as the gated one).
import os, time, numpy as np
from modelopt.onnx.quantization import quantize
W='/ws/experiments/compression/quantization/_work'
d=np.load(f'{W}/tokens_512_pruned.npz')   # REMAPPED ids
t0=time.time()
quantize(onnx_path=f'{W}/model_pruned.onnx',
         calibration_data={'input_ids':d['input_ids'][:512].astype(np.int64),'attention_mask':d['attention_mask'][:512].astype(np.int64)},
         output_path=f'{W}/model_int8_pruned.onnx', quantize_mode='int8', calibration_method='entropy',
         calibration_eps=['cuda:0'], op_types_to_quantize=['MatMul','Gemm'], high_precision_dtype='fp16')
print(f'QUANTIZE_DONE {time.time()-t0:.0f}s -> {os.path.getsize(f"{W}/model_int8_pruned.onnx")/1e6:.0f}MB', flush=True)
