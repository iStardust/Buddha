import numpy as np

if __name__ == "__main__":
    code_path = "checkpoint/buddha/exstyle_code.npy"
    ex_code = np.load(code_path,allow_pickle=True).item()
    print(len(ex_code.keys()))