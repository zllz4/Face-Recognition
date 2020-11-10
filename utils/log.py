import json
import visdom
import numpy as np


from easydict import EasyDict as edict

class Log(object):
    def __init__(self, name, backend="visdom"):
        super(Log, self).__init__()
        self.name = name
        self.backend = backend
        if self.backend == "visdom":
            self.vis = visdom.Visdom(env=self.name)
            self.vis.close()
        self.log = edict()

    def line(self, name):
        self.log[name] = edict()
        self.log[name].type = "line"
        self.log[name].x = []
        self.log[name].y = []

        # print(f"{name} is registered as a line")

    def bar(self, name):
        self.log[name] = edict()
        self.log[name].type = "bar"
        self.log[name].x = []
        self.log[name].y = []

        # print(f"{name} is registered as a bar")

    def append_point(self, name, x=None, y=None, delta_x=None, delta_y=None):
        '''
            append a point
        '''
        assert name in self.log.keys(), f"{name} have not been registered!"
        assert x is not None or delta_x is not None, "no x data!"
        assert y is not None or delta_y is not None, "no y data!"

        if x is not None:
            self.log[name].x.append(x)
        else:
            # print(len(self.log[name].x))
            if len(self.log[name].x) == 0:
                self.log[name].x.append(delta_x)
            else:
                self.log[name].x.append(self.log[name].x[-1]+delta_x)

        if y is not None:
            self.log[name].y.append(y)
        else:
            if len(self.log[name].y) == 0:
                self.log[name].y.append(delta_y)
            else:
                self.log[name].y.append(self.log[name].y[-1]+delta_y)

    def append_points(self, name, x, y):
        '''
            append many points
        '''

        assert name in self.log.keys(), f"{name} have not been registered!"
        self.log[name].x.append(x)
        self.log[name].y.append(y)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.log, f)
        print(f"[+] log save to {path}")

    def resume(self, path):
        with open(path, "r") as f:
            self.log = edict(json.load(f))
        print(f"[+] log resume from {path}")

    def display(self):
        for name in self.log.keys():
            if len(self.log[name].x) == 0:
                continue
            if self.log[name]['type'] == "line":
                if self.backend == "visdom":
                    self.vis.line(
                        X=np.array(self.log[name].x),
                        Y=np.array(self.log[name].y),
                        win=name,
                        name=name, 
                        opts=dict(
                            legend=[name],
                            title=name
                        )
                    )
            elif self.log[name]['type'] == "bar":

                if self.backend == "visdom":
                    self.vis.stem(
                        X=np.array(self.log[name].y[-1]),
                        Y=np.array(self.log[name].x[-1]),
                        win=name,
                        opts=dict(
                            legend=[name],
                            title=name
                        )
                    )
        