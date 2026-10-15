import os
import sys
import io

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import types
import numpy as np
import torch
import torch.nn.functional as F
import openunmix
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxconverter_common import float16
import onnx

def forward_dynamic(self, x):
    # x shape: (nb_samples, nb_channels, nb_bins, nb_frames)
    x = x.permute(3, 0, 1, 2)
    nb_frames, nb_samples, nb_channels, nb_bins = x.shape
    mix = x
    x = x[..., : self.nb_bins]
    x = (x + self.input_mean) * self.input_scale
    x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
    x = self.bn1(x)
    x = x.reshape(nb_frames, nb_samples, self.hidden_size)
    x = torch.tanh(x)
    lstm_out = self.lstm(x)
    x = torch.cat([x, lstm_out[0]], -1)
    x = self.fc2(x.reshape(-1, x.shape[-1]))
    x = self.bn2(x)
    x = F.relu(x)
    x = self.fc3(x)
    x = self.bn3(x)
    x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
    x = (x * self.output_scale + self.output_mean).clamp(min=0.0) * mix
    return x.permute(1, 2, 3, 0)

def main():
    os.makedirs("ai-spike/models", exist_ok=True)
    print("Loading umxhq vocals...")
    m = openunmix.umxhq(targets=['vocals'], pretrained=True).eval().target_models['vocals']
    m.forward = types.MethodType(forward_dynamic, m)
    
    dummy = torch.randn(1, 2, 2049, 100)
    out_fp32 = "ai-spike/models/openunmix_vocals_dynamic.onnx"
    print(f"Exporting dynamic ONNX to {out_fp32}...")
    torch.onnx.export(
        m,
        dummy,
        out_fp32,
        input_names=['spectrogram'],
        output_names=['filtered_spectrogram'],
        dynamic_axes={'spectrogram': {0: 'batch', 3: 'nb_frames'}, 'filtered_spectrogram': {0: 'batch', 3: 'nb_frames'}},
        opset_version=18
    )
    print("FP32 export completed!")
    
    # Quantize to INT8
    out_int8 = "ai-spike/models/openunmix_vocals_dynamic_int8.onnx"
    print(f"Quantizing to INT8: {out_int8}...")
    quantize_dynamic(out_fp32, out_int8, weight_type=QuantType.QUInt8)
    print("INT8 quantization completed!")
    
    # Convert to FP16
    out_fp16 = "ai-spike/models/openunmix_vocals_dynamic_fp16.onnx"
    print(f"Converting to FP16: {out_fp16}...")
    m_onnx = onnx.load(out_fp32)
    m16 = float16.convert_float_to_float16(m_onnx)
    onnx.save(m16, out_fp16)
    print("FP16 conversion completed!")
    
    # Test session with different frame counts
    sess = ort.InferenceSession(out_fp32, providers=['CPUExecutionProvider'])
    t1 = sess.run(None, {'spectrogram': np.random.randn(1, 2, 2049, 43).astype(np.float32)})[0]
    t2 = sess.run(None, {'spectrogram': np.random.randn(1, 2, 2049, 1292).astype(np.float32)})[0]
    print(f"Verification: out1={t1.shape}, out2={t2.shape}. Dynamic ONNX works perfectly!")

if __name__ == "__main__":
    main()
