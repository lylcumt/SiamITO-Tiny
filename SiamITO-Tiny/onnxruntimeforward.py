import os
import numpy as np
import time
import glob
import cv2
import torch
from PIL import Image
import onnxruntime as rt
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
import torchvision.transforms as transforms
import torch.nn.functional as F
from dataset import Pair
from dataset import Pair, center_crop
#

total = 0
mean_fps = 0
template_size = 21
search_size = 55
search_offset = int(search_size / 2)
center = search_offset + 1
COUNTER1 = 0

transform = transforms.ToTensor()
data_path = '../SiamITO1227/infrared_small_object_data/val'
seq_path = sorted(glob.glob(os.path.join(data_path, 'data*')))

hanning = np.hanning(search_size)
hanning = np.outer(hanning, hanning)
np.save('hanning', hanning)
for index in range(len(seq_path)):
    img_path = sorted(glob.glob(os.path.join(seq_path[index], '*')))
    anno_path = sorted(glob.glob(os.path.join(data_path, 'labels/*')))
    anno_file = open(anno_path[index], encoding='gbk')
    anno = []
    score = 0

    for line in anno_file:
        anno.append(line.strip())

    anno_template = anno[1].split('\t')

    x_t = int(anno_template[3])
    y_t = int(anno_template[4])

    template = np.array(cv2.imread(img_path[0], 0), dtype=np.uint8)
    template = center_crop(template, [template_size, template_size], x_t, y_t)
    template = transform(template)
    np.save('template.npy', template.numpy())
    # template = template.cuda()
    template = torch.unsqueeze(template, 0).detach().numpy()

    counter = 0
    time_start = time.time()
    for i in range(len(img_path)):


        o_search = np.array(cv2.imread(img_path[i], 0), dtype=np.uint8)
        search_region = center_crop(o_search, [search_size, search_size], x_t, y_t)
        search = transform(search_region)
        # search = search.cuda()
        numpy_array = search.numpy()
        np.save('search.npy', numpy_array)
        search = torch.unsqueeze(search, 0).detach().numpy()


        # sess = rt.InferenceSession("best_718_sim.onnx", None)
        sess = rt.InferenceSession("best_1127.onnx", None)  #best_718_sim_qt_PPQ.onnx0.8425  best_718_sim.onnx 0.878
                                                                                                            #best_824_sim_qt (3).onnx0.806   #best_sim_QT.onnx 0.8526
        input_name0 = sess.get_inputs()[0].name
        input_name1 = sess.get_inputs()[1].name
        out_name = sess.get_outputs()[0].name
        pred_onx = sess.run([out_name], {input_name0: template, input_name1: search})  # 执行推断
        pred_onx =torch.squeeze(torch.as_tensor(pred_onx),dim=1)
        pred_onx =torch.squeeze(pred_onx)
        pred_onx = torch.squeeze(pred_onx)


        prediction = pred_onx.cpu().detach().numpy()
        # prediction -= np.min(prediction)
        # prediction /= np.max(prediction)

        # prediction = hanning * prediction
        adaptiuve_boundary_suppression = False
        if adaptiuve_boundary_suppression:
            prediction_hann = hanning * prediction * hanning
            a, b = torch.topk(torch.as_tensor(prediction_hann).flatten(), k=1)  ##自适应汉宁窗抑制
            if a < 8:
                prediction = prediction
            else:
                prediction = prediction_hann
        else:
            prediction = hanning * prediction
        position = np.unravel_index(np.argmax(prediction), prediction.shape)

        COUNTER1+=1
        # if(COUNTER1%5==0):
        #     print(position)
        displace_x = position[1] - center + 1
        displace_y = position[0] - center + 1

        x_t += displace_x
        y_t += displace_y


        anno_search = anno[i + 1].split('\t')
        x_s = int(anno_search[3])
        y_s = int(anno_search[4])

        distance = np.sqrt((x_t - x_s) ** 2 + (y_t - y_s) ** 2)
        if distance < 8:
            score += 1 - distance / 8

        time_end = time.time()

        counter += 1

        o_search = cv2.cvtColor(o_search, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(o_search, (x_t - 7, y_t - 7), (x_t + 7, y_t + 7), (0, 0, 255), 1)
        # file_path = str(index)+".txt"
        # with open(file_path, 'a') as file:
        #     file.write(f"{x_t}, {y_t}\n")
        # cv2.putText(o_search, 'Fps:%f' % (1 / (time_end - time_start)), (0, 16),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.5, 0, 1)
        cv2.putText(o_search, 'C:%d' % (counter), (200, 16),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, 0, 1)

        if (index == 0):
            cv2.imwrite('./picture2/' + '%d.jpg' % (counter), o_search)
        if(index ==3):
            cv2.imwrite('./lunwentu/' + '%d.jpg' % (counter), o_search)
        cv2.imshow('test', o_search)
        # cv2.imshow('prediction', prediction)
        cv2.waitKey(1)
    timecost = time_end - time_start
    score = score / len(img_path)
    print('sequence %d score:%f' % (index + 1, score))
    print(f'sequence {index + 1} mean fps: {1 / (timecost / len(img_path))}')
    total += score
print('mean score %f',total/len(seq_path))
    #
