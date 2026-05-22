import os
import math

import numpy as np

video_frames = 97
vae_stride = 16
compressed_frames = video_frames // 4 + 1
spatial_compress_rate = os.environ['spatial_compress_rate']

def append_dummy_t(ratio2hws):
    for key in ratio2hws:
        for i in range(len(ratio2hws[key])):
            h, w = ratio2hws[key][i]
            ratio2hws[key][i] = (1, h, w)
    return ratio2hws

def get_full_ratio2hws(ratio2hws, total_pixels2scales):
    full_ratio2hws = {}
    for ratio, hws in ratio2hws.items():
        real_ratio = hws[-1][1] / hws[-1][2]
        full_ratio2hws[int(real_ratio*1000)/1000] = hws
        if ratio != 1.000:
            full_ratio2hws[int(1/real_ratio*1000)/1000] = [(item[0], item[2], item[1]) for item in hws]
    if 'heritage_' in spatial_compress_rate:
        dynamic_resolution_h_w = {}
        for ratio in full_ratio2hws:
            dynamic_resolution_h_w[ratio] = {}
            for pn, scale_times in total_pixels2scales.items():
                h, w = full_ratio2hws[ratio][-1][1] * scale_times, full_ratio2hws[ratio][-1][2] * scale_times
                # pixel = (h * vae_stride, w * vae_stride)
                scales = [(pt,scale_times*ph, scale_times*pw) for (pt, ph, pw) in full_ratio2hws[ratio]]
                dynamic_resolution_h_w[ratio][(h, w)] = {
                    'scales': scales,
                    'pn': pn,
                    'h_div_w': ratio,
                }
    else:
        dynamic_resolution_h_w = {}
        for ratio in full_ratio2hws:
            dynamic_resolution_h_w[ratio] = {}
            for pn, scales_num in total_pixels2scales.items():
                h, w = full_ratio2hws[ratio][scales_num-1][1], full_ratio2hws[ratio][scales_num-1][2]
                # pixel = (h * vae_stride, w * vae_stride)
                scales = full_ratio2hws[ratio][:scales_num]
                dynamic_resolution_h_w[ratio][(h, w)] = {
                    'scales': scales,
                    'pn': pn,
                    'h_div_w': ratio,
                }
    return dynamic_resolution_h_w

# ratio2hws = {
#     1.000: [(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(10,10),(12,12),(16,16),(24,24),(32,32),(48,48),(60,60),(64,64)],
#     1.250: [(1,1),(2,2),(3,3),(4,3),(5,4),(6,5),(7,5),(8,6),(10,8),(15,12),(20,16),(30,24),(35,28),(45,36),(66,52),(70,56)],
#     1.333: [(1,1),(2,2),(3,2),(4,3),(5,4),(6,5),(7,5),(8,6),(12,9),(16,12),(20,15),(28,21),(36,27),(48,36),(68,50),(72,54)],
#     1.500: [(1,1),(2,2),(3,2),(4,3),(5,3),(6,4),(7,4),(8,6),(12,8),(15,10),(21,14),(33,22),(39,26),(48,32),(72,48),(78,52)],
#     1.750: [(1,1),(2,2),(3,3),(4,3),(5,3),(6,4),(7,4),(8,5),(12,7),(14,8),(21,12),(32,18),(42,24),(54,30),(80,45),(84,48)],
#     2.000: [(1,1),(2,2),(3,2),(4,2),(5,3),(6,3),(7,4),(8,4),(12,6),(16,8),(22,11),(38,19),(46,23),(60,30),(82,41),(90,45)],
#     2.500: [(1,1),(2,2),(3,2),(4,2),(5,2),(7,3),(8,3),(10,4),(15,6),(20,8),(25,10),(40,16),(50,20),(65,26),(90,36),(100,40)],
#     3.000: [(1,1),(2,2),(3,2),(4,2),(5,2),(6,2),(8,3),(9,3),(15,5),(21,7),(27,9),(45,15),(54,18),(72,24),(96,32),(111,37)],
# }
# total_pixels2scales = {
#     '0.06M': 11,
#     '0.15M': 13,
#     '0.25M': 13,
#     '0.40M': 14,
#     '0.90M': 15,
#     '1M': 16,
# }

def get_ratio2hws_video_v2():
    import os
    if spatial_compress_rate == '16':
        ratio2hws_video_common_v2 = {}
        for h_div_w in [1, 100/116, 3/4, 2/3, 9/16, 1/2, 2/5, 1/3]:
            scale_schedule = []
            # 48*48 is 480p, 60*60 is 720p
            for scale in [1,2,3,4,5,6,7,8,10,12,16] + [24, 32, 40, 48, 60, 64]:
            # for scale in [4,8,12,20]:
                area = scale * scale
                pw_float = math.sqrt(area / h_div_w)
                ph_float = pw_float * h_div_w
                ph, pw = int(np.round(ph_float)), int(np.round(pw_float))
                scale_schedule.append((ph, pw))
            ratio2hws_video_common_v2[h_div_w] = scale_schedule
        total_pixels2scales = {
            # '16K': 8,
            '0.06M': 11,
            '0.25M': 13,
            '0.40M': 14,
            # '0.60M': 15,
            '0.92M': 16,
            '1M': 17,
        }
    elif spatial_compress_rate == '32':
        ratio2hws_video_common_v2 = {}
        for h_div_w in [1, 100/116, 3/4, 2/3, 9/16, 1/2, 2/5, 1/3]:
            scale_schedule = []
            # 48*48 is 480p, 60*60 is 720p
            # for scale in list(range(1,1+16)) + [20, 24, 30, 40]:
            for scale in [1,2,3,4,5,6,7,8] + [10,12,16,20,25,30]:
            # for scale in [4,8,12,20]:
                area = scale * scale
                pw_float = math.sqrt(area / h_div_w)
                ph_float = pw_float * h_div_w
                ph, pw = int(np.round(ph_float)), int(np.round(pw_float))
                scale_schedule.append((ph, pw))
            ratio2hws_video_common_v2[h_div_w] = scale_schedule
        total_pixels2scales = {
            '16K': 4,
            '0.06M': 8,
            '0.40M': 12,
        }
    elif spatial_compress_rate == 'heritage_schedule_stride_32':
        ratio2hws_video_common_v2 = {}
        for h_div_w in [1, 100/116, 3/4, 2/3, 9/16, 1/2, 2/5, 1/3]:
            scale_schedule = []
            # 48*48 is 480p, 60*60 is 720p
            # for scale in list(range(1,1+16)) + [20, 24, 30, 40]:
            for scale in [1,2,3,4,5,6,7,8]:
            # for scale in [4,8,12,20]:
                area = scale * scale
                pw_float = math.sqrt(area / h_div_w)
                ph_float = pw_float * h_div_w
                ph, pw = int(np.round(ph_float)), int(np.round(pw_float))
                scale_schedule.append((ph, pw))
            ratio2hws_video_common_v2[h_div_w] = scale_schedule
        total_pixels2scales = {
            '0.06M': 1,
            '0.25M': 2,
            '1M': 4,
        }
    elif spatial_compress_rate == 'heritage_schedule_stride_8':
        ratio2hws_video_common_v2 = {}
        for h_div_w in [1, 100/116, 3/4, 2/3, 9/16, 1/2, 2/5, 1/3]:
            scale_schedule = []
            # 48*48 is 480p, 60*60 is 720p
            # for scale in list(range(1,1+16)) + [20, 24, 30, 40]:
            for scale in [1,2,3,4,5,6,7,8]:
            # for scale in [4,8,12,20]:
                area = scale * scale
                pw_float = math.sqrt(area / h_div_w)
                ph_float = pw_float * h_div_w
                ph, pw = int(np.round(ph_float)), int(np.round(pw_float))
                scale_schedule.append((ph, pw))
            ratio2hws_video_common_v2[h_div_w] = scale_schedule
        total_pixels2scales = {
            '0.06M': 4,
            '0.25M': 8,
            '1M': 16,
        }
    return ratio2hws_video_common_v2, total_pixels2scales

ratio2hws, total_pixels2scales = get_ratio2hws_video_v2()
ratio2hws = append_dummy_t(ratio2hws)
dynamic_resolution_h_w = get_full_ratio2hws(ratio2hws, total_pixels2scales)
dynamic_resolution_thw = {}
for ratio in dynamic_resolution_h_w:
    for (h, w) in dynamic_resolution_h_w[ratio]:
        # spatial_time_schedule = []
        # spatial_time_schedule.extend(image_scale_schedule)
        # firstframe_scalecnt = len(image_scale_schedule)
        # if compressed_frames > 1:
        #     scale_schedule = dynamic_resolution_h_w[ratio][pn]['scales']
        #     # predefined_t = np.linspace(1, compressed_frames - 1, len(scale_schedule))
        #     predefined_t = np.linspace(1, compressed_frames - 1, total_pixels2scales['0.06M']-1).tolist() + [compressed_frames - 1] * (len(scale_schedule)-total_pixels2scales['0.06M']+1)
        #     spatial_time_schedule.extend([(min(int(np.round(predefined_t[i])), compressed_frames - 1), h, w) for i, (_, h, w) in enumerate(scale_schedule)])
        dynamic_resolution_thw[(h, w)] = dynamic_resolution_h_w[ratio][(h, w)]
        # dynamic_resolution_thw[(h, w)]['tower_split_index'] = firstframe_scalecnt
h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
# print(dynamic_resolution_thw)

if __name__ == '__main__':
    print(dynamic_resolution_thw)
    