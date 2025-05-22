import glob
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from dataset import center_crop

template_size = 25
search_size = 65
search_offset = int(search_size / 2)
center = search_offset + 1
hanning = np.hanning(search_size)
hanning = np.outer(hanning, hanning)

net = torch.load('./train_model/best_rand_template.pkl')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
net.eval()
transform = transforms.ToTensor()

seq_path = sorted(glob.glob('./real_video/*'))
cv2.namedWindow('show', 0)
# cv2.namedWindow('dst', 0)
clahe = cv2.createCLAHE(3, (8, 8))
#
# for index in range(len(seq_path)):
index = 2
img_paths = sorted(glob.glob(os.path.join(seq_path[index], '*')))
init = False
x_t, y_t = 0, 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 保存视频的编码
out = cv2.VideoWriter(f"./output/seq3.mp4", fourcc, 30.0, (640, 368))

for img_path in img_paths:
    key = cv2.waitKey(0)
    img = cv2.imread(img_path, 0)

    # img enhancement
    # dst = cv2.equalizeHist(img)
    dst = clahe.apply(img)
    # dst = img

    if key == ord('i') and not init:
        init_rect = cv2.selectROI('show', dst, False, False)
        init = True
        x_t = int(init_rect[0] + init_rect[2] / 2)
        y_t = int(init_rect[1] + init_rect[3] / 2)
        template = center_crop(dst, [template_size, template_size], x_t, y_t)
        template = transform(template)
        template = template.cuda()
        template = torch.unsqueeze(template, 0)
        template = net.conv(template)
        self_attn_template = net.self_attn(template)

    if init:
        search_region = center_crop(img, [search_size, search_size], x_t, y_t)
        search = transform(search_region)
        search = search.cuda()
        search = torch.unsqueeze(search, 0)
        search = net.conv(search)
        self_attn_search = net.self_attn(search)

        cross_attn_template = net.cross_attn(template, search)
        cross_attn_search = net.cross_attn(search, template)
        attn_template = torch.cat((self_attn_template, cross_attn_template), dim=1)
        attn_search = torch.cat((self_attn_search, cross_attn_search), dim=1)

        attn_template = net.adjust_attn(attn_template)
        attn_search = net.adjust_attn(attn_search)

        output = net.match_corr(attn_template, attn_search)
        output = torch.squeeze(output, 0)
        output = torch.squeeze(output, 0)

        prediction = output.cpu().detach().numpy()
        prediction -= np.min(prediction)
        prediction /= np.max(prediction)

        prediction = hanning * prediction
        position = np.unravel_index(np.argmax(prediction), prediction.shape)

        displace_x = position[1] - center + 1
        displace_y = position[0] - center + 1

        x_t += displace_x
        y_t += displace_y
        dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(dst, (x_t - 7, y_t - 7), (x_t + 7, y_t + 7), (0, 0, 255), 1)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img, (x_t - 7, y_t - 7), (x_t + 7, y_t + 7), (0, 0, 255), 1)

    cv2.imshow('show', dst)
    out.write(img)

