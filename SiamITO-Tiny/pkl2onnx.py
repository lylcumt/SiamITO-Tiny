import torch
import torch.onnx
from model import CrossAtt,SelfAtt,SiameseITO,Backbone
import onnx
from onnxsim import simplify
import os
import  onnxruntime

if __name__ == '__main__':

    model = torch.load('./train_model/best1127.pkl')
    model.eval().to('cuda')
    inputs = (torch.randn(1, 1, 21, 21).to('cuda'),torch.randn(1, 1, 55, 55).to('cuda'))
    torch.onnx.export(model, inputs, "best_1127.onnx", opset_version=11,verbose=True)
    print("模型转换成功!")

    idx_start =0
    onnx_model = onnx.load('best_1127.onnx')  # load onnx model
    for input in onnx_model.graph.input:
        for node in onnx_model.graph.node:
            # 如果当前节点的输入名称与待修改的名称相同，则将其替换为新名称
            for i, name in enumerate(node.input):
                if name == input.name:
                    node.input[i] = "input_" + str(idx_start)
        input.name = "input_" + str(idx_start)
        idx_start += 1

    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, 'best_1127_sim.onnx')
    print('finished exporting onnx')

    for name,x in model.named_parameters():

        print(name,x.shape,x)
    