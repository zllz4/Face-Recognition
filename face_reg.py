import visdom
import os
import time
import math
import json
import numpy as np
import datetime
import matplotlib.pyplot as plt
import struct
import torch
import torchvision
import torch.backends.cudnn as cudnn
from scipy import interpolate

def count_time(name):
    def decorator(func):
        def wrapper(*arg,**kw):
            start = time.time()
            func(*arg,**kw)
            print(f"< {name} end in {datetime.timedelta(seconds=(time.time()-start))} >")
        return wrapper
    return decorator

class FaceReg(object):
    '''
        A general face recognizer
    '''
    def __init__(self, CF):
        '''
            CF is a Config class
        '''

        CF.display()
        CF.generate()
        config = CF.config

        self.device = config.global_.device

        # dataset
        self.batch_size = config.train.batch_size
        self.test_batch_size = config.test.batch_size
        self.train_data_loader = config.train.loader
        self.train_transform = config.train.transform
        self.test_transform = config.test.transform
        self.classes = config.train.dataset.classes
        
        # model
        self.model = config.model.backbone
        self.margin_layer = config.model.loss
        self.model_save_dir = config.model.save_dir
        self.input_size = config.model.input_size

        # train & test
        self.optimizer = config.train.optimizer
        self.train_strategy = config.train.strategy
        self.test_dataset = config.test.dataset
        self.test_loader = config.test.loader

        # global
        self.name = config.global_.name

        # megaface
        self.megaface_enable = config.megaface.enable
        if self.megaface_enable:
            self.result_dir = config.megaface.result_save_dir
            self.test_no_noise = config.megaface.no_noise
            self.feature_save_dir = config.megaface.feature_save_dir
            self.devkit_dir = config.megaface.devkit_dir
            self.megaface_test_scale = config.megaface.scale
            self.probe = config.megaface.probe
            self.distractor = {}
            for s in self.megaface_test_scale:
                self.distractor[s] = config.megaface.distractor[str(s)]

        self.epoch = 0
        self.best_avg_acc = 0
        self.best_avg_acc_th = 0
        self.history = {}
        self.vis = visdom.Visdom(env=self.name)
        self.vis.close()
        self.set_eval()

    def _train(self, inputs, labels):
        features = self.model(inputs)
        loss, log = self.margin_layer(features, labels)
        loss.backward()
        self.optimizer.step()
        return loss, log

    def _test(self, inputs, labels):
        features = self.model(inputs)
        loss, log = self.margin_layer(features, labels)
        return loss, log

    def _predict(self, inputs):
        features = self.model(inputs)
        return features

    def set_train(self):
        '''
            Change the model to train state
        '''
        self.model.train()
        self.margin_layer.train()

    def set_eval(self):
        '''
            Change the model to evaluation state
        '''
        self.model.eval()
        self.margin_layer.eval()

    def save(self, save_dir=None):
        '''
            save model parameters to save dir

            Args:
                save_dir: dir where the model is saved
        '''
        if save_dir is None:
            save_dir = self.model_save_dir

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, "model.pth")

        torch.save({
            "model_state_dict": self.model.state_dict(), 
            "margin_layer_state_dict": self.margin_layer.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(), 
            "epoch": self.epoch, 
            "best_avg_acc": self.best_avg_acc,
            "best_avg_acc_th": self.best_avg_acc_th,
            "log": self.history
        }, save_path)

        print(f"[+] model save to {save_path}")

    def resume(self, resume_dir=None, resume_layer='all'):
        '''
            resume model parameters and train information from resume dir

            Args:
                resume_dir: dir where the model is saved, the saved data should be named as "model.pth"
                resume_layer:
                    "all": resume all layer
                    "feature extract": not resume the last margin layer
                    "log": just resume log
        '''

        if resume_dir is None:
            resume_dir = self.model_save_dir

        resume_path = os.path.join(resume_dir, "model.pth")
        checkpoint = torch.load(resume_path, map_location=torch.device(self.device))

        if resume_layer == 'feature extract':
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("load feature extract layer ")
        elif resume_layer == 'all':
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.margin_layer.load_state_dict(checkpoint["margin_layer_state_dict"])
            print("load feature extract layer + margin layer")
        elif resume_layer == 'log':
            pass
        else:
            raise RuntimeError(f"Invalid option {resume_layer}")
        # 这步有时候反而会使恢复的效果变差
        # self.optimizer.load_state_dict(model["optim_state_dict"])

        try:
            self.best_avg_acc = checkpoint["best_avg_acc"]
            self.best_avg_acc_th = checkpoint['best_avg_acc_th']
        except KeyError:
            print("can't find best_avg_acc or best_avg_acc_th")
        self.epoch = checkpoint["epoch"]
        self.history = checkpoint['log']
        
        print("epoch: ", self.epoch)
        print("best_avg_acc: ", self.best_avg_acc)
        print("best_avg_acc_th: ", self.best_avg_acc_th)
        print("optim lr:", self.optimizer.param_groups[0]["lr"])
        print(f"[+] model resume from {resume_path}")

    @count_time("val")
    def val(self, auto_save=False):
        '''
            val model
            Args:
                auto_save: if true, when current average accuracy is greater than best average accuracy, save current model automatically
        '''
        self.set_eval()

        print("> val <")
        print(f'{"name":<15}{"acc":<15}{"th":<15}{"time":<15}')

        acc_list = []
        th_list = []
        with torch.no_grad():
            for name in self.test_dataset.keys():
                dataset = self.test_dataset[name]
                loader = iter(self.test_loader[name])
                start = time.time()
                distance_all = np.array([])
                labels_all = np.array([])
                for i, data in enumerate(loader):
                    inputs, labels = data

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    inputs_fliplr = torch.flip(inputs, (3,))

                    features = self._predict(inputs)
                    features_fliplr = self._predict(inputs_fliplr)

                    features = torch.cat((features, features_fliplr), 1)

                    norm_features_1 = torch.nn.functional.normalize(features[::2]) 
                    norm_features_2 = torch.nn.functional.normalize(features[1::2])

                    distance = torch.sum(norm_features_1 * norm_features_2, 1)
                    distance_all = np.append(distance_all, torch.squeeze(distance).cpu().numpy())
                    labels_all = np.append(labels_all, labels[::2].cpu().numpy())

                    print(f"\r{i}/{len(loader)}", end="")
                best_acc = 0
                best_th = 0
                for th in np.sort(distance_all):
                    prediction_all = distance_all > th
                    result = (prediction_all == labels_all)

                    acc = np.mean(result.astype(np.float))
                    if acc > best_acc:
                        best_acc = acc
                        best_th = th
                acc_list.append(best_acc)
                th_list.append(best_th)
                print("\r                                                  ", end="")
                print(f"\r{name:<15}{best_acc:<15.8f}{best_th:<15.8f}{time.time()-start:<15}")
                self.log("{}_acc".format(name), dict(data_y=best_acc))

                # # for debug
                # false_index = np.nonzero(~((distance_all > best_th) == labels_all))
                # record_file = "false_index_"+name+".txt"
                # with open(record_file, "w") as f:
                #     f.write(str(np.shape(false_index)))
                #     f.write("\n".join([str(i) for i in false_index]))
                # # print(f"false index record in {record_file}")

        avg_acc = sum(acc_list)/len(acc_list)
        avg_th = sum(th_list)/len(th_list)
        if avg_acc > self.best_avg_acc:
            self.best_avg_acc = avg_acc
            self.best_avg_acc_th = avg_th
            if auto_save:
                self.save()
        self.display_log()

    @count_time("total train strategy")
    def train(self):
        '''
            train model, the function will call val() and train_one_epoch()
        '''
        train_epoch = 0
        for stage in self.train_strategy:
            for name, strategy in stage.items():
                if self.epoch > train_epoch + strategy['epoch']:
                    train_epoch += strategy['epoch']
                    print(f"jump stage {name}")
                    continue

                print(f"\n=> stage {name}, strategy {strategy}")
                train_epoch += strategy['epoch']
                while self.epoch < train_epoch:
                    self.train_one_epoch(strategy['lr'], strategy['log_interval'])
                    if strategy['test']:
                        self.val(auto_save=True)
        print("resume best model...")
        self.resume()
        self.val()          

    @count_time("train")
    def train_one_epoch(self, lr=None, log_interval=10):
        '''
            train the model for one epoch
            Args:
                lr: learning rate
                log_interval: the train steps between two log data
        '''
        
        # dont forget to set the model in train mode again！otherwise after one test step the model will remain in eval mode and bn layer will be locked forever, causing a huge drop in accuracy.
        self.set_train()
        
        if lr is not None:
            self.lr = lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        total_loss = 0
        total_true = 0
        total_num = 0
        total_cos_in = 0
        total_cos_out = 0
        theta_interval = 36
        total_theta_count = [0] * theta_interval

        log_loss = 0
        log_acc = 0

        start = time.time()
        for epoch in range(self.epoch+1, self.epoch+2):
            print("\n{}: epoch {}, lr {}".format(time.asctime(time.localtime(time.time())), epoch, str([param_group['lr'] for param_group in self.optimizer.param_groups])))
            print("> train <")
            for i, data in enumerate(self.train_data_loader, 0):
                # print(torch.utils.data.get_worker_info())
                inputs, labels = data

                # 必！须！要！梯度清零！
                self.optimizer.zero_grad()

                # to cuda
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # model run
                loss, log = self._train(inputs, labels)
                (outputs, target_cos_in, target_theta_in, target_cos_out) = log

                total_loss += loss.item()

                # compute acc
                _, pred = torch.max(outputs, 1)
                total_true += (pred == labels).float().sum().item()
                total_num += list(labels.size())[0]

                # trace target cos change
                total_cos_in += torch.mean(target_cos_in).item()
                total_cos_out += torch.mean(target_cos_out).item()
                for j in range(theta_interval):
                    total_theta_count[j] += torch.sum((math.pi/theta_interval*j <= target_theta_in) & (math.pi/theta_interval*(1+j) > target_theta_in)).item()
                # print(f"{i} {total_cos_out}")

                # monitoring status
                if (i+1) % log_interval == 0:
                    iter_per_s = log_interval / (time.time() - start)
                    print(f"\r{i+1}/{len(self.train_data_loader)} -> step {i+1}, loss {total_loss/log_interval:.8f}, acc {total_true / total_num:.4f}, speed {iter_per_s:.2f} iter/s      ", end="")

                    log_loss =  total_loss / log_interval
                    log_acc = total_true / total_num
                    log_cos_in = total_cos_in / log_interval
                    log_cos_out = total_cos_out / log_interval
                    log_theta = [i/sum(total_theta_count) for i in total_theta_count]

                    total_loss = 0
                    total_true = 0
                    total_num = 0
                    total_cos_in = 0
                    total_cos_out = 0
                    total_theta_count = [0]*theta_interval
                    start = time.time()

                    self.log("iter_per_s", dict(data_y=iter_per_s, delta_x=log_interval))
                    self.log("train_loss", dict(data_y=log_loss, delta_x=log_interval))
                    self.log("train_acc", dict(data_y=log_acc, delta_x=log_interval))
                    self.log("cos_in", dict(data_y=log_cos_in, delta_x=log_interval))
                    self.log("cos_out", dict(data_y=log_cos_out, delta_x=log_interval))
                    self.log("theta", dict(data_y=log_theta, data_x=np.linspace(0,180,theta_interval+1)[:-1]+(180/(theta_interval)/2), type="bar"))
                    self.display_log()
            else:
                log_interval_last = len(self.train_data_loader) % log_interval
                iter_per_s = log_interval_last / (time.time() - start)
                print(f"\r{i+1}/{len(self.train_data_loader)} -> step {i+1}, loss {total_loss/log_interval_last:.8f}, acc {total_true / total_num:.4f}, speed {iter_per_s:.2f} iter/s      ", end="")
                log_loss =  total_loss / log_interval_last
                log_acc = total_true / total_num
                log_cos_in = total_cos_in / log_interval_last
                log_cos_out = total_cos_out / log_interval_last
                log_theta = [i/sum(total_theta_count) for i in total_theta_count]

                self.log("iter_per_s", dict(data_y=iter_per_s,delta_x=log_interval_last))
                self.log("train_loss", dict(data_y=log_loss, delta_x=log_interval_last))
                self.log("train_acc", dict(data_y=log_acc, delta_x=log_interval_last))
                self.log("cos_in", dict(data_y=log_cos_in, delta_x=log_interval_last))
                self.log("cos_out", dict(data_y=log_cos_out, delta_x=log_interval_last))
                self.log("theta", dict(data_y=log_theta, data_x=np.linspace(0,180,theta_interval+1)[:-1]+(180/(theta_interval)/2), type="bar"))
                self.display_log()
                
            print()
            self.epoch = epoch
        
        self.set_eval()

    def log(self, name, data):
        '''
            name: name for log item
            data: dict
                type: data type

                --- type == "line" ---- (default)
                    new data point appends to the tail, will draw a growing line plot using the whole list of data_y
                    delta_x: data_x[-1] + delta_x will be appended to the tail of list data_x, default is 1
                    data_y: data_y will be appended to the tail of list data_y

                --- type == "bar" ----
                    new y data (a list) appends to the tail, new x data replace old ones, each item of data_y can be used to draw a whole plot
                    data_x: data_x will replace origin data_x
                    data_y: data_y (a new list) will be appended to the tail of list data_y
        '''
        if data.get("type", "line") == "line":
            if name in self.history.keys():
                self.history[name]['data_x'].append(self.history[name]['data_x'][-1] + data.get('delta_x', 1))
                self.history[name]['data_y'].append(data['data_y'])
                self.history[name]['type'] = "line"
            else:
                self.history[name] = {}
                self.history[name]['data_x'] = [data.get('delta_x', 1)]
                self.history[name]['data_y'] = [data['data_y']]
                self.history[name]['type'] = "line"
        elif data.get("type", "line") == "bar":
            if name in self.history.keys():
                self.history[name]['data_x'] = data.get('data_x', self.history[name]['data_x'])
                self.history[name]['data_y'].append(data['data_y'])
                self.history[name]['type'] = "bar"
            else:
                self.history[name] = {}
                self.history[name]['data_x'] = data['data_x']
                self.history[name]['data_y'] = [data['data_y']]
                self.history[name]['type'] = "bar"

    @count_time("test")
    def test_megaface(self):
        self.set_eval()

        assert os.path.isdir(self.feature_save_dir), f"{self.feature_save_dir} not exist!"
        print("> megaface test <")

        # gen feature
        self._megaface_gen_feature("facescrub_features", self.probe)
        print(" - probe set feature generate")
        for s in self.megaface_test_scale:
            self._megaface_gen_feature(f"megaface_{s}_features", self.distractor[s])
            print(f" - distractor set (scale {s}) feature generate")
        

        # megaface test
        print("-"*55)
        print(f'{"scale":<10}{"Identification":<15}{"Verification":<15}{"Time":<15}')
        for s in self.megaface_test_scale:
            print(f"\rrunning test (scale {s}) ...", end="")
            start = time.time()
            script_path = os.path.join(self.devkit_dir, "experiments/run_experiment_modified.py")
            distractor_feature_path = os.path.join(self.feature_save_dir, f"megaface_{s}_features")
            probe_feature_path = os.path.join(self.feature_save_dir, "facescrub_features")
            end = "_0.bin"
            result_dir = f"{self.result_dir}/megaface_result_{s}"
            no_noise = "-nn" if self.test_no_noise else ""
            if not os.path.isdir(result_dir):
                os.makedirs(result_dir)
            os.system(f"python {script_path} {distractor_feature_path} {probe_feature_path} {end} {result_dir} -s {s} -d {no_noise}> {os.path.join(result_dir, 'log')}")
            idn, ver = self._megaface_draw_result(os.path.join(result_dir, f"cmc_facescrub_megaface_0_{s}_1.json"), result_dir)
            print(f"\r{s:<10}{idn:<15.8f}{ver:<15.8f}{time.time()-start:<15.8f}")
        print("-"*55)
            # os.system('python run_experiment.py -p /data/share/megaface/devkit/templatelists/facescrub_uncropped_features_list.json /home/yujie/Desktop/face/data/test/megaface/megaface_images_aligned/ /home/yujie/Desktop/face/data/test/megaface/facescrub_images/ _0.bin /home/yujie/Desktop/face/src/megaface/results -s 1000000')

        # for debug
        # def read_feature(filename):
        #     f = open(filename, 'rb')
        #     rows, cols, stride, type_ = struct.unpack('iiii', f.read(4 * 4))
        #     mat = np.fromstring(f.read(rows * 4), dtype=np.dtype('float32'))
        #     return print(mat.reshape(rows, 1)[1:10])
        # read_feature("/home/yujie/Desktop/face/data/test/megaface/facescrub_images/Michael_Landes/Michael_Landes_43667.png_0.bin")
        # read_feature("/home/yujie/Desktop/face/megaface/facescrub_features/Michael_Landes/Michael_Landes_43667.png_0.bin")
                
        # self._megaface_draw_result('megaface/results/cmc_facescrub_megaface_0_1000000_1.json')
        # gen_feature('/data/share/megaface/facescrub_images/', self, '/home/yujie/Desktop/face/data/test/megaface/facescrub_images/')
        # gen_feature('/data/share/megaface/megaface_images_aligned/', self, '/home/yujie/Desktop/face/data/test/megaface/megaface_images_aligned/')

    def _megaface_gen_feature(self, out_dir, dataset):
        self.set_eval()
        
        end = "_0.bin"

        if not os.path.isdir(os.path.join(self.feature_save_dir, out_dir)):
            os.mkdir(os.path.join(self.feature_save_dir, out_dir))

        print(f"delete .bin file in {os.path.join(self.feature_save_dir, out_dir)}")
        os.system(f'find {os.path.join(self.feature_save_dir, out_dir)} -name "*.bin" -type f -delete')
        
        with torch.no_grad():
            loader = dataset.loader
            with open(dataset.file) as f:
                path = [line for line in json.load(f)['path']]
                # print(path)
                # print(self.probe['file'])
                # raise
            for i, data in enumerate(iter(loader)):
                inputs, labels = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                features = self._predict(inputs)

                norm_features = torch.nn.functional.normalize(features).cpu().numpy()

                # if i==0:
                #     print(inputs.size())
                #     print(inputs[0][0][0][1:10])
                #     print(features[0][1:10])
                #     print(norm_features[0][1:10])
                #     print(path[0])

                for j in range(len(norm_features)):
                    idx = i * self.test_batch_size + j
                    file_path = os.path.join(self.feature_save_dir, out_dir, path[idx]) + end
                    if not os.path.isdir(os.path.dirname(file_path)):
                        os.makedirs(os.path.dirname(file_path))
                    with open(file_path, "wb") as f:
                        f.write(struct.pack('iiii', norm_features.shape[1], 1, 4, 5))
                        # print(np.shape(norm_features))
                        f.write(norm_features[j].data)
                    # print(f"\r{idx+1}/{len(path)} write image feature {file_path}                   ",end="")
                    print(f"\r{idx+1}/{len(path)}",end="")


    def _megaface_draw_result(self, result_path, out_dir):
        with open(result_path) as f:
            result = json.load(f)

        plt.close('all')

        # draw cmc
        plt.figure()
        fig, axes = plt.subplots()
        axes.set_xlabel('Rank')
        axes.set_ylabel('Identification Rate %')
        axes.set_xscale('log')
        axes.set_title(f"Identification @ 1e6 distractors Rank 1 = {result['cmc'][1][0]}")
        axes.plot(np.array(result['cmc'][0])+1, result['cmc'][1])
        fig.savefig(os.path.join(out_dir, 'cmc.jpg'))

        # draw roc
        # roc = interpolate.interp1d(result['roc'][0], result['roc'][1], kind='cubic')
        roc = interpolate.interp1d(result['roc'][0], result['roc'][1], kind='linear')
        plt.figure()
        fig, axes = plt.subplots()
        axes.plot(result['roc'][0], result['roc'][1])
        axes.set_title(f'Verification @ 1e-6 = {roc(1e-6)}')
        axes.set_xlabel('False Positive Rate')
        axes.set_ylabel('True Positive Rate')
        fig.savefig(os.path.join(out_dir, 'roc.jpg'))

        return result['cmc'][1][0], roc(1e-6)

    def display_log(self):
        '''
            send log data to visdom server
        '''
        # first draw doesn't need replace

        for item in self.history.keys():
            if self.history[item]['type'] == "line":
                self.vis.line(
                    X=np.array(self.history[item]['data_x']),
                    Y=np.array(self.history[item]['data_y']),
                    win=item,
                    name=item, 
                    opts=dict(
                        legend=[item],
                        title=item
                    )
                )
            elif self.history[item]['type'] == "bar":
                self.vis.stem(
                    X=np.array(self.history[item]['data_y'][-1]),
                    Y=np.array(self.history[item]['data_x']),
                    win=item,
                    opts=dict(
                        legend=[item],
                        title=item
                    )
                )

        # if len(self.history['train_epoch']) == 1:
        #     self.vis.line(
        #         X=np.array(range(len(self.history['train_loss']))),
        #         Y=np.array(self.history["train_loss"]),
        #         win="loss",
        #         name="train_loss", 
        #         opts=dict(
        #             legend=["train_loss", "test_loss"],
        #             title="loss"
        #         )
        #     )
        #     self.vis.line(
        #         X=np.array(range(len(self.history['train_acc']))),
        #         Y=np.array(self.history["train_acc"]),
        #         win="acc",
        #         name="train_acc", 
        #         opts=dict(
        #             legend=["train_acc", "test_acc"],
        #             title="acc"
        #         )
        #     )
        # else:
        #     self.vis.line(
        #         X=np.array(range(len(self.history['train_loss']))),
        #         Y=np.array(self.history["train_loss"]),
        #         win="loss",
        #         name="train_loss",
        #         update="replace"
        #     )
        #     self.vis.line(
        #         X=np.array(range(len(self.history['train_acc']))),
        #         Y=np.array(self.history["train_acc"]),
        #         win="acc",
        #         name="train_acc", 
        #         update="replace"
        #     )

        # self.vis.line(
        #     X=np.array(range(len(self.history['test_loss']))),
        #     Y=np.array(self.history["test_loss"]),
        #     win="loss",
        #     name="test_loss",
        #     update="replace"
        # )

        # self.vis.line(
        #     X=np.array(range(len(self.history['test_acc']))),
        #     Y=np.array(self.history["test_acc"]),
        #     win="acc",
        #     name="test_acc",
        #     update="replace"
        # )
