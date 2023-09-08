import argparse

import colour
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src_path', type=str)
    parser.add_argument('dst_path', type=str)
    args = parser.parse_args()

    lut = torch.load(args.src_path)["LUT"]["LUT"]
    lut = colour.LUT3D(lut.clip(0.0, 1.0).permute(3, 2, 1, 0).numpy())
    colour.write_LUT(lut, args.dst_path)
