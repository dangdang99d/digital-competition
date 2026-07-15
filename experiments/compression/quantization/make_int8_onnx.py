# Shippable int8 Q/DQ ONNX: fp32 input (uniform types, RAM fits on 15GB box) + fp16 high-precision
# (non-quant parts fp16 -> ~627MB output, fits 1GB cap) + MatMul/Gemm-only (prevents F1 collapse).
import sys, os, time, numpy as np
from modelopt.onnx.quantization import quantize
W='/ws/experiments/compression/quantization/_work'
d=np.load(f'{W}/tokens_512.npz')
t0=time.time()
quantize(onnx_path=f'{W}/model.onnx',
         calibration_data={'input_ids':d['input_ids'][:512].astype(np.int64),'attention_mask':d['attention_mask'][:512].astype(np.int64)},
         output_path=f'{W}/model_int8_ship.onnx', quantize_mode='int8', calibration_method='entropy',
         calibration_eps=['cuda:0'], op_types_to_quantize=['MatMul','Gemm'], high_precision_dtype='fp16')
print(f'QUANTIZE_DONE {time.time()-t0:.0f}s -> {os.path.getsize(f"{W}/model_int8_ship.onnx")/1e6:.0f}MB', flush=True)
