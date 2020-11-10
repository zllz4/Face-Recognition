from config import Config
from face_reg import FaceReg

def webface():
    CF = Config()
    config = CF.config
    FR = FaceReg(CF)
    FR.train()

def main():
    webface()

if __name__ == "__main__":
    main()
