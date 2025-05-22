import os
import numpy as np
import time

import torch
from PIL import Image
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
import torchvision.transforms as transforms
import torch.nn.functional as F
from dataset import Pair

datapath = '../SiamITO1227/infrared_small_object_data/val'
transform = transforms.ToTensor()
dataset = Pair(datapath, transforms=transform, rand_choice=True,
                search_size=55, template_size=25, r_pos=6)
calibration_dataloader = [
{
    'input_0':dataset.__getitem__(0)[0].reshape([1, 1, 25, 25]).detach().numpy(),
    'input_1':dataset.__getitem__(0)[1].reshape([1, 1, 55, 55]).detach().numpy()
}
    for _ in range(512)
]

# template = np.random.randn(1, 1, 25, 25).astype(np.float32)
template = calibration_dataloader[0]['input_0']
# search = np.random.randn(1, 1, 55, 55).astype(np.float32)
search = calibration_dataloader[1]['input_1']

# print(calibration_dataloader)
class DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder, augmented_model_path=None):
        self.image_folder = calibration_image_folder
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False

            self.enum_data_dicts = iter(calibration_dataloader)
        return next(self.enum_data_dicts, None)

def main():
    input_model_path = 'best_0106_sim.onnx'  # 输入onnx模型
    output_model_path = 'best_0106_qt.onnx'  # 输出模型名
    calibration_dataset_path = 'D:/SiamITO(random)/infrared_small_object_data/train/data04'  # 校准数据集图像地址
    # 用于校准数据加载,注意这个方法里面需要做图像一些操作,与pytorch训练的时候加载数据操作一致
    dr = DataReader(calibration_dataset_path, input_model_path)
    # 开始量化
    quantize_static(input_model_path,
                    output_model_path,
                    dr,
                    quant_format=QuantFormat.QDQ,
                    per_channel=False,
                    weight_type=QuantType.QInt8)
    print("量化完成")



if __name__ == "__main__":
    main()
# #
# import onnxmltools
# from onnxmltools.utils.float16_converter import convert_float_to_float16
# input_onnx_model = 'best_0106.onnx'
# # Change this path to the output name and path for your float16 ONNX model
# output_onnx_model = 'best_0106_f16.onnx'
# # Load your model
# onnx_model = onnxmltools.utils.load_model(input_onnx_model)
# # Convert tensor float type from your input ONNX model to tensor float16
# onnx_model = convert_float_to_float16(onnx_model)
# # Save as protobuf
# onnxmltools.utils.save_model(onnx_model, output_onnx_model)