import os
import json
import torch
import torch.backends.cudnn as cudnn

from datetime import datetime
from torchvision import transforms
from easydict import EasyDict as edict

from models import loss
from models import resnet
from models import resnet_ir
from dataset import dataset

class Config(object):
    def __init__(self):
        config = edict()

        # global
        config.global_ = edict()
        config.global_.name = "default" # config name, also visdom env name
        config.global_.gpu = "0"
        # config.global_.use_metric_learning_loss = False
        config.global_.result_dir = datetime.now().strftime("%m_%d_%H_%M_%S")
        config.global_.auto_clear = True # auto clear generate file
        config.global_.device = None # to be generated

        # model
        config.model = edict()
        config.model.input_size = (1, 112, 112) # (channel, width, height)
        config.model.backbone = "resnet_18_ir"
        config.model.feature_dim = 512
        config.model.loss = "arcloss"
        config.model.loss_param = edict(s=32, m=0.5)
        config.model.save_dir = "./checkpoint/model/"

        # train
        config.train = edict()
        config.train.dataset = edict(name="webface", path="H:/dataset/face/train/casia_webface_insight")
        config.train.strategy = [
                edict({"stage1": {"lr":1e-1, "epoch":10, "test":True, "log_interval":10}}),
                edict({"stage2": {"lr":1e-2, "epoch":10, "test":True, "log_interval":10}}),
                edict({"stage3": {"lr":1e-3, "epoch":10, "test":True, "log_interval":10}})
            ]
        config.train.batch_size = 64
        config.train.transform = edict()
        config.train.transform.random_horizontal_flip = True
        config.train.transform.normalize = True
        config.train.optimizer = "SGD"
        config.train.weight_decay = 5e-4
        config.train.momentum = 0.9

        # test
        config.test = edict()
        config.test.batch_size = 64 # currently useless
        config.test.dataset = [
            edict(name="lfw", path="H:/dataset/face/test/lfw/test_pair.txt"),
            edict(name="cplfw", path="H:/dataset/face/test/cplfw/test_pair.txt"),
            edict(name="calfw", path="H:/dataset/face/test/calfw/test_pair.txt"),
            edict(name="agedb-30", path="H:/dataset/face/test/agedb-30/test_pair.txt"),
            edict(name="cfp-ff", path="H:/dataset/face/test/cfp-ff/test_pair.txt"),
            edict(name="cfp-fp", path="H:/dataset/face/test/cfp-fp/test_pair.txt"),
        ]

        # megaface
        config.megaface = edict()
        config.megaface.enable = False # if you dont want to do the megaface test, just change it to false
        config.megaface.no_noise = True # whether to remove noise pictures in megaface dataset
        config.megaface.devkit_dir = 'megaface/devkit'
        config.megaface.megaface_dataset_dir = '/data/share/megaface/megaface_images_aligned'
        config.megaface.facescrub_dataset_dir = '/data/share/megaface/facescrub_images'
        config.megaface.feature_save_dir = os.path.join("tmp",config.global_.result_dir,"megaface_feature")
        config.megaface.clear_feature = True # auto clear the saved feature after megaface test to release space
        config.megaface.result_save_dir = os.path.join("tmp",config.global_.result_dir,"megaface_result")
        config.megaface.clear_result = True # auto clear the saved result after megaface test to release space
        config.megaface.scale = [str(10**i) for i in range(1,7)]

        self.config = config
    
    def _print_title(self, title):
        print()
        print(f"{title.upper():^35}")
        print("-"*35)

    def display(self):
        # print("="*75)
        # print(f"{'CONFIG:'+self.config.global_.name:^75}")
        # print("="*75)
        # print("-"*100)
        print("=> display config")
        
        for top_key in self.config.keys():
            # print("="*75)
            # print()
            # print(f"{top_key.upper():^35}")
            # print("-"*35)
            self._print_title(top_key)
            for key in self.config[top_key].keys():
                if type(self.config[top_key][key]) == list and type(self.config[top_key][key][0]) == edict:
                    print(f"{key+':':<25}")
                    for item in self.config[top_key][key]:
                        print(" "*25 + f"{item}")
                else:
                    print(f"{key+':':<25}{self.config[top_key][key]}")
        print()

    def generate(self):
        print("=> generate config")
        
        self._print_title("global")
        # print("global "+"-"*15)
        self.delete_list = []

        if not os.path.isdir(self.config.model.save_dir):
            os.makedirs(self.config.model.save_dir)
        with open(os.path.join(self.config.model.save_dir, "config.txt"), "w") as f:
            json.dump(self.config, f, indent=2)
        print("config saved in " + os.path.join(self.config.model.save_dir, "config.txt"))
        # assert not os.listdir(self.config.model.save_dir), f"{self.config.model.save_dir} is not empty!"

        if not os.path.isdir(os.path.join("tmp",self.config.global_.result_dir)):
            os.makedirs(os.path.join("tmp",self.config.global_.result_dir))
        assert not os.listdir(os.path.join("tmp",self.config.global_.result_dir)), f"{os.path.join('tmp',self.config.global_.result_dir)} is not empty!"

        # print()
        self._print_title("gpu")
        # print("gpu "+"-"*15)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config.global_.gpu
        print("Use", torch.cuda.device_count(), "GPUs!")

        if torch.cuda.is_available():
            self.config.global_.device = "cuda"
            cudnn.benchmark = True
            print("cuda available, set config.global_.device to cuda")
        else:
            self.config.global_.device = "cpu"
            print("cuda not available, set config.global_.device to cpu")
        
        # print()
        self._print_title("transform")
        # print("transform "+"-"*15)
        train_transform_list = [transforms.Resize(self.config.model.input_size[1:])]
        if self.config.train.transform['random_horizontal_flip']:
            train_transform_list.append(transforms.RandomHorizontalFlip())
        train_transform_list.extend([transforms.Grayscale(),transforms.ToTensor()])
        if self.config.train.transform['normalize']:
            train_transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        train_transform = transforms.Compose(train_transform_list)
        print(f"train_transform:\n{train_transform}")

        test_transform_list = [transforms.Resize(self.config.model.input_size[1:])]
        test_transform_list.extend([transforms.Grayscale(),transforms.ToTensor()])
        if self.config.train.transform['normalize']:
            test_transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        test_transform = transforms.Compose(test_transform_list)
        print(f"test_transform:\n{test_transform}")
        self.config.train.transform = train_transform
        self.config.test.transform = test_transform

        # print()
        self._print_title("dataset")
        # print("dataset "+"-"*15)
        image_list, class_name_map = dataset.list_all_image(self.config.train.dataset.path, f"tmp/{self.config.global_.result_dir}/webface.txt")
        train_dataset = dataset.Dataset(image_list, class_name_path=class_name_map, transform=self.config.train.transform)
        train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.train.batch_size, shuffle=True)
        print(f"train_dataset: {train_dataset}")
        print(f"image_list file path: {image_list}")
        print(f"class_name_map file path: {class_name_map}")
        self.config.train.dataset = train_dataset
        self.config.train.loader = train_dataset_loader
        self.delete_list.extend([image_list, class_name_map])

        test_dataset_dict = edict()
        test_loader_dict = edict()
        for test_pair in self.config.test.dataset:
            test_dataset = dataset.Dataset(test_pair['path'], transform=self.config.test.transform)
            test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config.test.batch_size, shuffle=False)
            test_dataset_dict[test_pair['name']] = test_dataset
            test_loader_dict[test_pair['name']] = test_dataset_loader
            print(f"{test_pair['name']}: {test_dataset}")
        self.config.test.dataset = test_dataset_dict
        self.config.test.loader = test_loader_dict

        # print()
        self._print_title("model")
        # print("model "+"-"*15)

        if self.config.model.backbone == "resnet_18_ir":
            self.config.model.backbone = resnet_ir.ResNet18_IR(self.config.model.input_size, self.config.model.feature_dim)
        elif self.config.model.backbone == "resnet_34_ir":
            self.config.model.backbone = resnet_ir.ResNet34_IR(self.config.model.input_size, self.config.model.feature_dim)
        elif self.config.model.backbone == "resnet_50_ir":
            self.config.model.backbone = resnet_ir.ResNet50_IR(self.config.model.input_size, self.config.model.feature_dim)
        else:
            raise RuntimeError(f"Invalid Backbone Option {self.config.model.backbone}")
        self.config.model.backbone.to(self.config.global_.device)
        if self.config.global_.device == "cuda":
            self.config.model.backbone = torch.nn.DataParallel(self.config.model.backbone)
        
        with open(os.path.join(self.config.model.save_dir, "model_structure.txt"), "w") as f:
            f.write(str(self.config.model.backbone))
        print("model structure saved in " + os.path.join(self.config.model.save_dir, "model_structure.txt"))
        # self.delete_list.append(f"tmp/{self.config.global_.result_dir}/model_structure.txt")

        if self.config.model.loss == "arcloss":
            self.config.model.loss = loss.ArcLoss(self.config.model.feature_dim, len(self.config.train.dataset.classes), **self.config.model.loss_param)
        elif self.config.model.loss == "cosloss":
            self.config.model.loss = loss.CosLoss(self.config.model.feature_dim, len(self.config.train.dataset.classes), **self.config.model.loss_param)
        elif self.config.model.loss == "mixloss":
            self.config.model.loss = loss.MixLoss(self.config.model.feature_dim, len(self.config.train.dataset.classes), **self.config.model.loss_param)
        elif self.config.model.loss == "sphereloss":
            self.config.model.loss = loss.SphereLoss(self.config.model.feature_dim, len(self.config.train.dataset.classes), **self.config.model.loss_param)
        elif self.config.model.loss == "normsoftmaxsloss":
            self.config.model.loss = loss.NormSoftmaxLoss(self.config.model.feature_dim, len(self.config.train.dataset.classes), **self.config.model.loss_param)
        elif self.config.model.loss == "softmaxsloss":
            self.config.model.loss = loss.SoftmaxLoss(self.config.model.feature_dim, len(self.config.train.dataset.classes), **self.config.model.loss_param)
        else:
            raise RuntimeError(f"Invalid Loss Option {self.config.model.loss}")
        
        self.config.model.loss.to(self.config.global_.device)
        if self.config.global_.device == "cuda":
            self.config.model.loss = torch.nn.DataParallel(self.config.model.loss)
        
        print(f"Loss: {self.config.model.loss}")
        print(f"model param {sum([p.numel() for p in self.config.model.backbone.parameters()])/1024/1024:.8f}M margin param {sum([p.numel() for p in self.config.model.loss.parameters()])/1024/1024:.8f}M")
        
        # print()
        self._print_title("optimizer")
        # print("optimizer "+"-"*15)
        # print("="*18+f">{'optimizer':^12}<"+"="*18)
        if self.config.train.optimizer == "SGD":
            optimizer = torch.optim.SGD([{'params': self.config.model.backbone.parameters()}, {'params': self.config.model.loss.parameters()}], lr=1e-5, weight_decay=self.config.train.weight_decay, momentum=self.config.train.momentum)
        else:
            raise RuntimeError(f"Invalid Optimizer Option {self.config.train.optimizer}")
        print(f"optimizer: {optimizer}")
        self.config.train.optimizer = optimizer

        if self.config.megaface.enable:

            # print()
            self._print_title("megaface")
            # print("megaface "+"-"*15)
            # print("="*18+f">{'megaface':^12}<"+"="*18)
            
            assert os.path.isdir(self.config.megaface.devkit_dir), f"{self.config.megaface.devkit_dir} not exist or is not a directory!"
            assert os.path.isdir(self.config.megaface.megaface_dataset_dir), f"{self.config.megaface.megaface_dataset_dir} not exist or is not a directory!"
            assert os.path.isdir(self.config.megaface.facescrub_dataset_dir), f"{facescrub_dataset_dir} not exist or is not a directory!"
            if not os.path.isdir(self.config.megaface.feature_save_dir):
                os.makedirs(self.config.megaface.feature_save_dir)
            if not os.path.isdir(self.config.megaface.result_save_dir):
                os.makedirs(self.config.megaface.result_save_dir)
            assert not os.listdir(self.config.megaface.feature_save_dir), f"{self.config.megaface.feature_save_dir} is not empty!"
            assert not os.listdir(self.config.megaface.result_save_dir), f"{self.config.megaface.result_save_dir} is not empty!"

            # probe
            if self.config.megaface.no_noise:
                probe_list_file = os.path.join(self.config.megaface.devkit_dir, "templatelists/facescrub_features_list_no_noise.json")
            else:
                probe_list_file = os.path.join(self.config.megaface.devkit_dir, "templatelists/facescrub_features_list.json")
            
            probe_file = open(f"tmp/{self.config.global_.result_dir}/probe.txt", 'w')
            with open(probe_list_file) as f:
                probe_list = json.load(f)['path']
                # print(f"open proble list file, total {len(probe_list)} probe images")
                for line in probe_list:
                    probe_file.write(os.path.join(self.config.megaface.facescrub_dataset_dir, line) + '\n')
            probe_file.close()
            
            self.delete_list.append(f"tmp/{self.config.global_.result_dir}/probe.txt")
            
            probe_dataset = dataset.Dataset(f"tmp/{self.config.global_.result_dir}/probe.txt", transform=self.config.test.transform)
            print(len(probe_dataset))
            probe_dataset_loader = torch.utils.data.DataLoader(probe_dataset, batch_size=self.config.test.batch_size, shuffle=False)
            print(len(probe_dataset_loader))
            probe = edict()
            probe.file = probe_list_file
            probe.dataset = probe_dataset
            probe.loader = probe_dataset_loader
            self.config.megaface.probe = probe
            print(f"probe dataset: {self.config.megaface.probe.dataset}")
            
            # distractor
            self.config.megaface.distractor = edict()
            for s in self.config.megaface.scale:
                if self.config.megaface.no_noise:
                    distractor_list_file = os.path.join(self.config.megaface.devkit_dir, f"templatelists/megaface_features_list_no_noise.json_{s}_1")
                else:
                    distractor_list_file = os.path.join(self.config.megaface.devkit_dir, f"templatelists/megaface_features_list.json_{s}_1")
                distractor_file = open(f"tmp/{self.config.global_.result_dir}/distractor_{s}.txt", 'w')
                with open(distractor_list_file) as f:
                    distractor_list = json.load(f)['path']
                    # print(f"open distractor list file scale {s}, total {len(distractor_list)} distractor images")
                    for line in distractor_list:
                        distractor_file.write(os.path.join(self.config.megaface.megaface_dataset_dir, line) + '\n')
                distractor_file.close()
                
                self.delete_list.append(f"tmp/{self.config.global_.result_dir}/distractor_{s}.txt")
                
                distractor_dataset = dataset.Dataset(f"tmp/{self.config.global_.result_dir}/distractor_{s}.txt", transform=self.config.test.transform)
                distractor_dataset_loader = torch.utils.data.DataLoader(distractor_dataset, batch_size=self.config.test.batch_size, shuffle=False)
                distractor = edict()
                distractor.file = distractor_list_file
                distractor.dataset = distractor_dataset
                distractor.loader = distractor_dataset_loader
                self.config.megaface.distractor[str(s)] = distractor
                print(f"distractor dataset scale {s}: {distractor_dataset}")
        
        # print()
        self._print_title("delete list")
        # print("deleta list "+"-"*15)
        # print("="*17+f">{'delete list':^14}<"+"="*17)
        for line in self.delete_list:
            print(line)
            if self.config.global_.auto_clear:
                os.remove(line)
        print("Remark: The generated files above will be auto cleared if set")
                
        
        print()
# config = Config()
# config.display()
# config.generate()
