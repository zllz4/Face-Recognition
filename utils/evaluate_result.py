import os
import numpy as np
from config import Config
from face_reg import FaceReg
import matplotlib.pyplot as plt

class Webface_Config(Config):
    name = "evaluate"
    model = "resnet-18-ir"
    margin_layer = "arcmargin"
    margin_layer_param = dict(s=30, m=0.4)
    gpu = ""
    batch_size = 128
    classes_select = [1]
    model_save_path = "../checkpoint/" + name + ".pth"

config = Webface_Config()
fr = FaceReg(config)

target_name = [
    "backbone_r18_arc_0.5.pth",
    # "param_r18_arc_0.3.pth",
    # "param_r18_arc_0.1.pth",
    "margin_r18_linear.pth"
]

cmp_names = ['cos_out','cos_in']
fig, ax_array = plt.subplots(1,2,figsize=(14,6))
for i,cmp_name in enumerate(cmp_names):
    axes = ax_array[i]
    for name in target_name:
        fr.resume(os.path.join("../checkpoint/saved_model",name), "log")
        len_of_points = len(fr.history[cmp_name]['data_y'])
        delete_index = []
        for i in range(len_of_points):
            if (i - 383) % 384 == 0:
                delete_index.append(i)
        data_x = np.delete(np.array(fr.history[cmp_name]['data_x']), delete_index)
        data_y = np.delete(np.array(fr.history[cmp_name]['data_y']), delete_index)
        axes.plot(data_x, data_y)
        axes.set_xlabel("iteration")
        axes.set_ylabel("logits")
    axes.legend(['arc_0.5','softmax'])
    axes.set_title(cmp_name)
fig.savefig("cmp_result.jpg")

# target_name = [
#     "param_r18_cos_0.7.pth",
#     "margin_r18_cos_0.35.pth",
#     "param_r18_cos_0.3.pth",
#     "param_r18_cos_0.1.pth",
# ]

# cmp_name = 'theta'
# fig, ax_array = plt.subplots(len(target_name),1,figsize=(8,16))
# name_list = ['m=0.7','m=0.35','m=0.3','m=0.1']
# for i,name in enumerate(target_name):

#     fr.resume(os.path.join("../checkpoint/saved_model",name), "log")
#     data_x = fr.history[cmp_name]['data_x']
#     data_y = fr.history[cmp_name]['data_y'][-1]
    
#     markerline1, stemlines1, baseline1 = ax_array[i].stem(data_x, data_y)
#     plt.setp(stemlines1, color=np.random.rand(3), linewidth=1)
#     markerline1.set_markersize(1)
#     baseline1.set_lw(0.1)

#     data_y = fr.history[cmp_name]['data_y'][0]
#     markerline2, stemlines2, baseline2 = ax_array[i].stem(data_x, data_y)
#     plt.setp(stemlines2, color=np.random.rand(3), linewidth=1)
#     markerline2.set_markersize(0)
#     baseline2.set_lw(0.1)
    
#     ax_array[i].set_xlabel("θ")
#     ax_array[i].set_ylabel("logits")
#     ax_array[i].legend([name_list[i]])
# fig.tight_layout()
# # axes.legend(['arc','cos','softmax','sphere','sphere_arc','norm'])
# fig.savefig("cmp_result.jpg")