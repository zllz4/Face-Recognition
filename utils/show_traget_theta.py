import math
import numpy as np
import matplotlib.pyplot as plt 

theta = np.linspace(0, math.pi, 500)

def softmax(theta):
    return np.cos(theta)

def arcface(theta, m):
    return np.cos(theta+m)

def arcface_md(theta, m):
    th = math.cos(math.pi-m)
    mm = math.sin(math.pi-m) * m
    m_cos = math.cos(m)
    m_sin = math.sin(m)
    cos = np.cos(theta)
    sin = np.sqrt(1-cos**2)
    return np.where(np.cos(theta)>th, cos*m_cos-sin*m_sin, cos-mm)

def arcface_md2(theta, m):
    th = math.cos(math.pi-m)
    mm = math.cos(math.pi-m) + 1
    m_cos = math.cos(m)
    m_sin = math.sin(m)
    cos = np.cos(theta)
    sin = np.sqrt(1-cos**2)
    return np.where(np.cos(theta)>th, cos*m_cos-sin*m_sin, cos-mm)

def cosface(theta, m):
    return np.cos(theta) - m

def sphereface(theta, m, lda):
    k = theta // (math.pi / m)
    # print(k)
    return (np.power(-1, k) * np.cos(theta * m) - 2*k + lda * np.cos(theta)) / (1+lda)

theta_degree = theta / math.pi * 180
plt.figure()
plt.title("target logit curve")
plt.xlabel("θ")
plt.ylabel("logits")

# plt.plot(theta_degree, arcface(theta, 0.5))
# plt.plot(theta_degree, arcface_md(theta, m=0.5))
# plt.plot(theta_degree, arcface_md2(theta, m=0.5))
# plt.legend(['origin arcface', 'msin(pi-m)', '1+cos(pi-m)'])

plt.plot(theta_degree, softmax(theta))
plt.plot(theta_degree, arcface(theta, 0.5))
plt.plot(theta_degree, cosface(theta, 0.35))
# plt.plot(theta_degree, sphereface(theta, m=4, lda=5))
# plt.plot(theta_degree, sphereface(theta, m=1.5, lda=0))
# plt.plot(theta_degree, sphereface(theta, m=2, lda=0))
# plt.legend(["softmax", "arcface m=0.5", "cosface m=0.35", "sphereface m=4 λ=5", "sphereface m=1.35 λ=0", "sphereface m=2 λ=0"])
plt.legend(["softmax", "arcface m=0.5", "cosface m=0.35"])
plt.savefig('target_logits.jpg')
