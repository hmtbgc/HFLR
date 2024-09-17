import argparse


def get_args():
    parser = argparse.ArgumentParser(description='GraphSAINT')
    parser.add_argument("--dataset", type=str, help="Name of dataset.")

    parser.add_argument("--num_roots", type=int, default=3000,
                        help="Expected number of sampled root nodes when using random walk sampler")

    parser.add_argument("--lr", type=float, default=0.005, help="Initial learning rate")
    parser.add_argument("--wd", type=float, default=0, help="Weight decay")
    parser.add_argument("--epoch", type=int, default=50, help="Maximum training epoch")
    parser.add_argument("--every_val", type=int, default=2, help="# epoch for each validation")
    parser.add_argument("--n_layer", type=int, default=3, help="# layer of GNN, including input and output layer")
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--num_repeat", type=int, default=50)
    parser.add_argument("--hid", type=int, default=256, help="Hidden feature size")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout ratio")
    parser.add_argument("--memory", type=int, default=0)

    args = parser.parse_args()
    return args
