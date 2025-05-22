import model
from dataset import Pair, center_crop

import os
import glob
import random
import cv2
import numpy as np
import time

import torch
import torchvision.transforms as transforms
import  matplotlib.pyplot as plt

def val(val_data_path, val_model, save_video=False, vis=False):
    print("***Begin to val model***")

    total = 0
    mean_fps = 0
    template_size = 25
    search_size = 55
    search_offset = int(search_size/2)
    center = search_offset + 1

    net = val_model

    transform = transforms.ToTensor()
    data_path = val_data_path
    seq_path = sorted(glob.glob(os.path.join(data_path, 'data*')))

    hanning = np.hanning(search_size)
    hanning = np.outer(hanning, hanning)

    for index in range(2,len(seq_path)):
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
        template = template.cuda()
        template = torch.unsqueeze(template, 0)
        template = net.conv(template)
        self_attn_template = net.self_attn(template)

        if vis:
            cv2.namedWindow('test', 0)
            # cv2.namedWindow('feature', 0)
            # cv2.namedWindow('search_region', 0)
            # cv2.namedWindow('prediction', 0)
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
            out = cv2.VideoWriter(f"./output/output{index}.avi", fourcc, 30.0, (256, 256))

        time_start = time.time()
        counter=0
        for i in range(len(img_path)):
            counter+=1
            o_search = np.array(cv2.imread(img_path[i], 0), dtype=np.uint8)
            search_region = center_crop(o_search, [search_size, search_size], x_t, y_t)

            # if vis:
            #     cv2.imshow('search', search)
            #     cv2.waitKey(0)

            search = transform(search_region)
            search = search.cuda()
            search = torch.unsqueeze(search, 0)
            search = net.conv(search)
            self_attn_search = net.self_attn(search)

            # cross_attn_template = net.cross_attn(template, search)
            cross_attn_search = net.cross_attn(search, template)
            # attn_template = torch.cat((self_attn_template, cross_attn_template), dim=1)
            # attn_search = torch.cat((self_attn_search, cross_attn_search), dim=1)

            # attn_template = net.adjust_attn(attn_template)
            # attn_search = net.adjust_attn(attn_search)

            output = net.match_corr(template, cross_attn_search)

            output = torch.squeeze(output, 0)
            output = torch.squeeze(output, 0)

            prediction = output.cpu().detach().numpy()
            # prediction -= np.min(prediction)
            # prediction /= np.max(prediction)


            prediction = hanning * prediction
            position = np.unravel_index(np.argmax(prediction), prediction.shape)

            displace_x = position[1] - center + 1
            displace_y = position[0] - center + 1

            x_t += displace_x
            y_t += displace_y

            anno_search = anno[i+1].split('\t')
            x_s = int(anno_search[3])
            y_s = int(anno_search[4])

            distance = np.sqrt((x_t-x_s)**2 + (y_t-y_s)**2)
            if distance < 8:
                score += 1 - distance/8

            if vis:
                o_search = cv2.cvtColor(o_search, cv2.COLOR_GRAY2BGR)

                cv2.rectangle(o_search, (x_t-7, y_t-7), (x_t+7, y_t+7), (0, 0, 255), 1)
                cv2.rectangle(o_search, (x_t - 28, y_t - 28), (x_t + 28, y_t + 28), (255    , 0, 0), 1)
                cv2.putText(o_search, 'C:%d' % (counter), (200, 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, 0, 1)

                cv2.imshow('test', o_search)
                cv2.imshow('prediction', prediction)

            response_map_normalized = cv2.normalize(prediction, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            response_map_normalized = response_map_normalized.astype(np.uint8)
            heatmap = cv2.applyColorMap(response_map_normalized, cv2.COLORMAP_JET)
            cv2.imshow("Heatmap", heatmap)

            k = cv2.waitKey(0)
            if k == 27:  # 按ESC退出
                cv2.destroyAllWindows()
            elif k == ord('s'):  # 按s保存并退出
                cv2.imwrite('with'+str(counter)+'.jpg', heatmap)
                cv2.imwrite('res'+str(counter)+'.jpg', o_search)
                # cv2.destroyAllWindows()
            if save_video:
                out.write(o_search)
            # print(f'sequence{index}_{i}')
            cv2.waitKey(1)

        time_end = time.time()
        mean_fps = time_end - time_start
        score = score / len(img_path)
        print('sequence %d score:%f' % (index+1, score))
        print(f'sequence {index+1} mean fps: {1 / (mean_fps/len(img_path))}')
        total += score
    return total / len(seq_path)


if __name__ == '__main__':
    val_data = './infrared_small_object_data/val'

    device = torch.device("cuda")
    val_net = torch.load('./train_model/best0106.pkl')
    val_net.to(device)
    val_net.eval()

    total_score = val(val_data, val_net, vis=True)
    print(f'mean score: {total_score}')

    model = val_net
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total: {total_num}, Trainable: {trainable_num}')
