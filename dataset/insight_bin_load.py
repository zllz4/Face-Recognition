import os
import cv2
import pickle
from PIL import Image
import numpy as np
import mxnet as mx

def insight_bin_load(bin_path, out_dir, align_size=(112,112)):
    '''
        load face in insight face xxx.bin file
    '''
    bins, issame_list = pickle.load(open(bin_path, 'rb'), encoding='bytes')
    # print(issame_list)
    print(len(issame_list))
    print(len(bins))
    test_pair = []
    for i in range(len(bins)):
        # if i > 100:
        #     break
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = Image.fromarray((img).astype(np.uint8))
        img = img.resize(align_size)
        save_path = os.path.join(out_dir, "{}.jpg".format(i+1))
        img.save(save_path)
        print("{}/{} save image {} to {}".format((i+1), len(bins), "{}.jpg".format(i+1), save_path))
        if (i+1) % 2 == 0:
            test_pair.append("{} {}\n{} {}".format(os.path.join(out_dir, f"{i}.jpg"), int(issame_list[i//2]), os.path.join(out_dir, f"{i+1}.jpg"), int(issame_list[i//2])))
        # img.show()
        # input()
    with open(os.path.join(out_dir, "test_pair.txt"), "w") as f:
        f.write("\n".join(test_pair))

def insight_rec_load(rec_path, idx_path, out_dir, align_size=(112,112)):
    '''
        load face in insight face xxx.rec file
    '''

    rec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
    info = rec.read_idx(0)
    header, _ = mx.recordio.unpack(info)
    max_idx = int(header.label[0])
    for idx in range(1, max_idx):
        img_info = rec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        # img.show()
        # raise
        img = img.resize(align_size)
        if not os.path.isdir(os.path.join(out_dir, str(label))):
            os.mkdir(os.path.join(out_dir, str(label)))
        save_path = os.path.join(out_dir, str(label), f"{idx}.jpg")
        img.save(save_path)
        print("{}/{} save image {} to {}".format(idx, max_idx, "{}.jpg".format(idx), save_path))
