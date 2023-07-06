
import torch
import numpy as np
import torch.nn as nn
import cv2


KV_filter = np.array([[-1,2,-2,2,-1],
                      [2,-6,8,-6,2],
                      [-2,8,-12,8,-2],
                      [2,-6,8,-6,2],
                      [-1,2,-2,2,-1]], dtype=np.float32)

LoG_filter = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 2, 1, 0],
                       [1, 2,-16,2, 1],
                       [0, 1, 2, 1, 0],
                       [0, 0, 1, 0, 0]], dtype=np.float32)


KB3_filter = np.array([
        [-1, 2, -1],
        [2, -4, 2],
        [-1, 2, -1]
    ], dtype=np.float32)

Lap_filter = np.array([
        [0, 1,  0],
        [1, -4, 1],
        [0, 1,  0]
    ], dtype=np.float32)

LAP2_filter = np.array([
        [1, 1,  1],
        [1, -8, 1],
        [1, 1,  1]
    ], dtype=np.float32)

LAP3_filter = np.array([
        [1, 4,  1],
        [4, -20, 4],
        [1, 4,  1]
    ], dtype=np.float32)


HP7_filter = np.array([[1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, -24, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1]], dtype=np.float32)


HP8_filter = np.array([[0, -1, -1, -1, 0],
                       [-1, 2, -4, 2, -1],
                       [-1,-4, 20,-4, -1],
                       [-1, 2, -4, 2, -1],
                       [0, -1, -1, -1, 0]], dtype=np.float32)


LPF_filter = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.float32) / 9.0


LPF2_filter = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.float32) / 8.0


def Sobel_filters():
    filter_1 = np.array([
        [1,  0, -1],
        [2,  0, -2],
        [1,  0, -1]
        ], dtype=np.float32)

    filter_2 = np.array([
        [1,  2,  1],
        [0,  0,  0],
        [-1, -2, -1]
        ], dtype=np.float32) 

    Sobel = np.array([filter_1, filter_2])
    return Sobel


def Prewitt_filters():
    filter_1 = np.array([
        [1,  0, -1],
        [1,  0, -1],
        [1,  0, -1]
        ], dtype=np.float32)

    filter_2 = np.array([
        [1,  1,  1],
        [0,  0,  0],
        [-1, -1, -1]
        ], dtype=np.float32) 

    Prewitt = np.array([filter_1, filter_2])
    return Prewitt


def LAP2_KV5():
    filter_1 = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1,  1, 0],
            [0, 1, -8, 1, 0],
            [0, 1, 1,  1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)

    filter_2  = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1]
    ], dtype=np.float32)

    all_hpf_list = np.array([filter_1, filter_2])

    return all_hpf_list




def hpf8_filters():
    KB3_filter = np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32)

    Lap_filter = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1,  0, 0],
            [0, 1, -4, 1, 0],
            [0, 0, 1,  0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)

    LAP2_filter = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1,  1, 0],
            [0, 1, -8, 1, 0],
            [0, 1, 1,  1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)

    LAP3_filter = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 4,  1, 0],
            [0, 4, -20, 4, 0],
            [0, 1, 4,  1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)

    KV_filter = np.array([[-1,2,-2,2,-1],
                      [2,-6,8,-6,2],
                      [-2,8,-12,8,-2],
                      [2,-6,8,-6,2],
                      [-1,2,-2,2,-1]], dtype=np.float32)

    LoG_filter = np.array([[0, 0, 1, 0, 0],
                        [0, 1, 2, 1, 0],
                        [1, 2,-16,2, 1],
                        [0, 1, 2, 1, 0],
                        [0, 0, 1, 0, 0]], dtype=np.float32)

    HP7_filter = np.array([[1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, -24, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1]], dtype=np.float32)

    HP8_filter = np.array([[0, -1, -1, -1, 0],
                       [-1, 2, -4, 2, -1],
                       [-1,-4, 20,-4, -1],
                       [-1, 2, -4, 2, -1],
                       [0, -1, -1, -1, 0]], dtype=np.float32)

    all_hpf_list = np.array([KB3_filter, Lap_filter, LAP2_filter, LAP3_filter,  KV_filter,  LoG_filter, HP7_filter, HP8_filter])

    return all_hpf_list


def hpf6_filters():
    filter_1 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, -1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)

    filter_2 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)
    

    filter_3 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, -2, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)

    filter_4 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)
    

    filter_5 = np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32)

    filter_6 = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1]
    ], dtype=np.float32)

    all_hpf_list = np.array([filter_1, filter_2, filter_3, filter_4, filter_5, filter_6])

    return all_hpf_list


def srm_filters():
    filter_class_1 = [
        np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 0]
        ], dtype=np.float32),
        np.array([
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 0]
        ], dtype=np.float32),
        np.array([
            [0, 0, 1],
            [0, -1, 0],
            [0, 0, 0]
        ], dtype=np.float32),
        np.array([
            [0, 0, 0],
            [1, -1, 0],
            [0, 0, 0]
        ], dtype=np.float32),
        np.array([
            [0, 0, 0],
            [0, -1, 1],
            [0, 0, 0]
        ], dtype=np.float32),
        np.array([
            [0, 0, 0],
            [0, -1, 0],
            [1, 0, 0]
        ], dtype=np.float32),
        np.array([
            [0, 0, 0],
            [0, -1, 0],
            [0, 1, 0]
        ], dtype=np.float32),
        np.array([
            [0, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
    ]

    filter_class_2 = [
        np.array([
            [1, 0, 0],
            [0, -2, 0],
            [0, 0, 1]
        ], dtype=np.float32),
        np.array([
            [0, 1, 0],
            [0, -2, 0],
            [0, 1, 0]
        ], dtype=np.float32),
        np.array([
            [0, 0, 1],
            [0, -2, 0],
            [1, 0, 0]
        ], dtype=np.float32),
        np.array([
            [0, 0, 0],
            [1, -2, 1],
            [0, 0, 0]
        ], dtype=np.float32),
    ]

    filter_class_3 = [
        np.array([
            [-1, 0, 0, 0, 0],
            [0, 3, 0, 0, 0],
            [0, 0, -3, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32),
        np.array([
            [0, 0, -1, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, -3, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32),
        np.array([
            [0, 0, 0, 0, -1],
            [0, 0, 0, 3, 0],
            [0, 0, -3, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32),
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -3, 3, -1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32),
        np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, -3, 0, 0],
            [0, 0, 0, 3, 0],
            [0, 0, 0, 0, -1]
        ], dtype=np.float32),
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, -3, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, -1, 0, 0]
        ], dtype=np.float32),
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, -3, 0, 0],
            [0, 3, 0, 0, 0],
            [-1, 0, 0, 0, 0]
        ], dtype=np.float32),
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [-1, 3, -3, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)
    ]

    filter_edge_3x3 = [
        np.array([
            [-1, 2, -1],
            [2, -4, 2],
            [0, 0, 0]
        ], dtype=np.float32),
        np.array([
            [0, 2, -1],
            [0, -4, 2],
            [0, 2, -1]
        ], dtype=np.float32),
        np.array([
            [0, 0, 0],
            [2, -4, 2],
            [-1, 2, -1]
        ], dtype=np.float32),
        np.array([
            [-1, 2, 0],
            [2, -4, 0],
            [-1, 2, 0]
        ], dtype=np.float32),
    ]

    filter_edge_5x5 = [
        np.array([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32),
        np.array([
            [0, 0, -2, 2, -1],
            [0, 0, 8, -6, 2],
            [0, 0, -12, 8, -2],
            [0, 0, 8, -6, 2],
            [0, 0, -2, 2, -1]
        ], dtype=np.float32),
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ], dtype=np.float32),
        np.array([
            [-1, 2, -2, 0, 0],
            [2, -6, 8, 0, 0],
            [-2, 8, -12, 0, 0],
            [2, -6, 8, 0, 0],
            [-1, 2, -2, 0, 0]
        ], dtype=np.float32),
    ]

    square_3x3 = np.array([
        [-1, 2, -1],
        [2, -4, 2],
        [-1, 2, -1]
    ], dtype=np.float32)

    square_5x5 = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1]
    ], dtype=np.float32)

    all_hpf_list = filter_class_1 + filter_class_2 + filter_class_3 + filter_edge_3x3 + filter_edge_5x5 + [square_3x3,
                                                                                                           square_5x5]

    hpf_3x3_list = filter_class_1 + filter_class_2 + filter_edge_3x3 + [square_3x3]
    hpf_5x5_list = filter_class_3 + filter_edge_5x5 + [square_5x5]

    normalized_filter_class_2 = [hpf / 2 for hpf in filter_class_2]
    normalized_filter_class_3 = [hpf / 3 for hpf in filter_class_3]
    normalized_filter_edge_3x3 = [hpf / 4 for hpf in filter_edge_3x3]
    normalized_square_3x3 = square_3x3 / 4
    normalized_filter_edge_5x5 = [hpf / 12 for hpf in filter_edge_5x5]
    normalized_square_5x5 = square_5x5 / 12

    all_normalized_hpf_list = filter_class_1 + normalized_filter_class_2 + normalized_filter_class_3 + \
                              normalized_filter_edge_3x3 + normalized_filter_edge_5x5 + [normalized_square_3x3,
                                                                                         normalized_square_5x5]
    all_hpf_list_5x5 = []

    for hpf_item in all_normalized_hpf_list:
        if hpf_item.shape[0] == 3:
            hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
        all_hpf_list_5x5.append(hpf_item)
    normalized_hpf_3x3_list = filter_class_1 + normalized_filter_class_2 + normalized_filter_edge_3x3 + [
        normalized_square_3x3]
    normalized_hpf_5x5_list = normalized_filter_class_3 + normalized_filter_edge_5x5 + [normalized_square_5x5]

    normalized_3x3_list = normalized_filter_edge_3x3 + [normalized_square_3x3]
    normalized_5x5_list = normalized_filter_edge_5x5 + [normalized_square_5x5]

    return all_hpf_list_5x5


class HPF_KB3(nn.Module):
    def __init__(self):
        super(HPF_KB3, self).__init__()

        filter_list = [KB3_filter]
        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 3, 3), requires_grad=False)
        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = input / 255.0
        output = self.hpf(output)

        return output

class HPF_Lap(nn.Module):
    def __init__(self):
        super(HPF_Lap, self).__init__()

        filter_list = [Lap_filter]
        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 3, 3), requires_grad=False)
        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = input / 255.0
        output = self.hpf(output)

        return output

class HPF_LAP2(nn.Module):
    def __init__(self):
        super(HPF_LAP2, self).__init__()

        filter_list = [LAP2_filter]
        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 3, 3), requires_grad=False)
        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = input / 255.0
        output = self.hpf(output)

        return output

class HPF_SRM1(nn.Module):
    def __init__(self):
        super(HPF_SRM1, self).__init__()

        filter_list = [KV_filter]
        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 5, 5), requires_grad=False)
        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=5, padding=2, bias=False, padding_mode='replicate')
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = input / 255.0
        output = self.hpf(output)

        return output


class HPF_SRM30(nn.Module):
    def __init__(self):
        super(HPF_SRM30, self).__init__()

        filter_list = srm_filters()
        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 5, 5), requires_grad=False)
        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = input / 255.0
        output = self.hpf(output)
        #output = self.tlu(output)

        return output



class HPF_srm6(nn.Module):
    def __init__(self):
        super(HPF_srm6, self).__init__()

        filter_list = hpf6_filters()

        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=5, padding=2, bias=False, padding_mode='replicate')
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = self.hpf(input)
        return output

class HPF_SRM6(nn.Module):
    def __init__(self):
        super(HPF_SRM6, self).__init__()

        filter_list = hpf6_filters()
        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 5, 5), requires_grad=False)
        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=5, padding=2, bias=False, padding_mode='replicate')
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = input / 255.0
        output = self.hpf(output)
        return output


class HPF_8HPFs(nn.Module):
    def __init__(self):
        super(HPF_8HPFs, self).__init__()

        filter_list = hpf8_filters()
        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 5, 5), requires_grad=False)
        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=5, padding=2, bias=False, padding_mode='replicate')
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = input / 255.0
        output = self.hpf(output)
        return output


class HPF_Sob(nn.Module):
    def __init__(self):
        super(HPF_Sob, self).__init__()

        filter_list = Sobel_filters()
        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 3, 3), requires_grad=False)
        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = input / 255.0
        output = self.hpf(output)
        return output


class HPF_Prew(nn.Module):
    def __init__(self):
        super(HPF_Prew, self).__init__()

        filter_list = Prewitt_filters()
        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 3, 3), requires_grad=False)
        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = input / 255.0
        output = self.hpf(output)
        return output


class HPF_LoG(nn.Module):
    def __init__(self):
        super(HPF_LoG, self).__init__()

        filter_list = [LoG_filter]
        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 5, 5), requires_grad=False)
        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=5, padding=2, bias=False, padding_mode='replicate')
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = input / 255.0
        output = self.hpf(output)

        return output

class HPF_LAP3(nn.Module):
    def __init__(self):
        super(HPF_LAP3, self).__init__()

        filter_list = [LAP3_filter]
        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 3, 3), requires_grad=False)
        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = input / 255.0
        output = self.hpf(output)

        return output

class HPF_HP7(nn.Module):
    def __init__(self):
        super(HPF_HP7, self).__init__()

        filter_list = [HP7_filter]
        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 5, 5), requires_grad=False)
        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=5, padding=2, bias=False, padding_mode='replicate')
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = input / 255.0
        output = self.hpf(output)

        return output


class HPF_HP8(nn.Module):
    def __init__(self):
        super(HPF_HP8, self).__init__()

        filter_list = [HP8_filter]
        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 5, 5), requires_grad=False)
        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=5, padding=2, bias=False, padding_mode='replicate')
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = input / 255.0
        output = self.hpf(output)

        return output


class HPF_LPF(nn.Module):
    def __init__(self):
        super(HPF_LPF, self).__init__()

        filter_list = [LPF_filter]
        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 3, 3), requires_grad=False)
        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = input
        output = self.hpf(output)

        return output


class HPF_LPF2(nn.Module):
    def __init__(self):
        super(HPF_LPF2, self).__init__()

        filter_list = [LPF2_filter]
        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 3, 3), requires_grad=False)
        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = input
        output = self.hpf(output)

        return output



class HPF_LAP2KV5(nn.Module):
    def __init__(self):
        super(HPF_LAP2KV5, self).__init__()

        filter_list = LAP2_KV5()
        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 5, 5), requires_grad=False)
        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=5, padding=2, bias=False, padding_mode='replicate')
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = input / 255.0
        output = self.hpf(output)
        return output




P_filter = np.array([[0,0,5.2,0,0],
                     [0,23.4,36.4,23.4,0],
                     [5.2,36.4,-261,36.4,5.2],
                     [0,23.4,36.4,23.4,0],
                     [0,0,5.2,0,0]], dtype=np.float32)/261

def gabor_filters():
    filters = []
    ksize = [5]
    lamda = np.pi / 2.0
    sigma = [0.5, 1.0]
    phi = [0, np.pi / 2]
    #    filters.append(KV_filter)
    #    filters.append(P_filter)
    # all_normalized_hpf_list = srm_filters()
    # for hpf_item in all_normalized_hpf_list:
    #     row_1 = int((5 - hpf_item.shape[0]) / 2)
    #     row_2 = int((5 - hpf_item.shape[0]) - row_1)
    #     col_1 = int((5 - hpf_item.shape[1]) / 2)
    #     col_2 = int((5 - hpf_item.shape[1]) - col_1)
    #     hpf_item = np.pad(hpf_item, pad_width=((row_1, row_2), (col_1, col_2)), mode='constant')
    #     filters.append(hpf_item)
    #filters = srm_filters()
    for theta in np.arange(0, np.pi, np.pi / 8):  # gabor 0 22.5 45 67.5 90 112.5 135 157.5
        for k in range(2):
            for j in range(2):
                kern = cv2.getGaborKernel((ksize[0], ksize[0]), sigma[k], theta, sigma[k] / 0.56, 0.5, phi[j],
                                          ktype=cv2.CV_32F)
                # print(1.5*kern.sum())
                # kern /= 1.5*kern.sum()
                filters.append(kern)
    return filters


class HPF_P(nn.Module):
    def __init__(self):
        super(HPF_P, self).__init__()

        filter_list = [P_filter]

        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        #self.tlu = TLU(5.0)

        # self.sc_bn_1 = nn.BatchNorm2d(30)

        # nn.init.constant_(self.sc_bn.weight, 1.0)

    def forward(self, input):
        output = self.hpf(input)
        #output = self.tlu(output)

        return output

class HPF_gabor32(nn.Module):
    def __init__(self):
        super(HPF_gabor32, self).__init__()

        filter_list = gabor_filters()

        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        #self.tlu = TLU(5.0)

        # self.sc_bn_1 = nn.BatchNorm2d(30)

        # nn.init.constant_(self.sc_bn.weight, 1.0)

    def forward(self, input):
        output = self.hpf(input)
        #output = self.tlu(output)

        return output

class HPF_spg62(nn.Module):
    def __init__(self):
        super(HPF_spg62, self).__init__()

        filter_list = srm_filters() + gabor_filters()

        hpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, len(filter_list), kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        #self.tlu = TLU(5.0)

        # self.sc_bn_1 = nn.BatchNorm2d(30)

        # nn.init.constant_(self.sc_bn.weight, 1.0)

    def forward(self, input):
        output = self.hpf(input)
        #output = self.tlu(output)

        return output



