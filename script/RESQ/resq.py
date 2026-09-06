import numpy as np
import struct
import time
import os
from utils import *
import argparse
from tqdm import tqdm

source = './DATA'


def Orthogonal(D):
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)
    return Q


def GenerateBinaryCode(X, P):
    XP = np.dot(X, P)
    binary_XP = (XP > 0)
    X0 = np.sum(XP * (2 * binary_XP - 1) / D ** 0.5, axis=1, keepdims=True) / np.linalg.norm(XP, axis=1, keepdims=True)
    return binary_XP, X0


def PackBinaryCodes(binary_codes):
    n, bits = binary_codes.shape
    if bits % 32 != 0:
        raise ValueError(f'RESQ code width must be a multiple of 32, got {bits}')
    storage_bits = (bits + 63) // 64 * 64
    padded_codes = np.pad(binary_codes, ((0, 0), (0, storage_bits - bits)), 'constant')
    packed_codes = np.packbits(padded_codes.reshape(-1, 8, 8)[:, ::-1]).view(np.uint64)
    return packed_codes.reshape(n, storage_bits // 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='gist')
    parser.add_argument('-b', '--bits', help='quantized bits', default=128)
    args = vars(parser.parse_args())
    dataset = args['dataset']
    bits = int(args['bits'])
    # path
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_proj.fvecs')

    C = 4096
    centroids_path = os.path.join(path, f'p{dataset}_centroid_{C}.fvecs')
    dist_to_centroid_path = os.path.join(path, f'p{dataset}_dist_to_centroid_{C}.fvecs')
    cluster_id_path = os.path.join(path, f'p{dataset}_cluster_id_{C}.ivecs')

    X = read_fvecs(data_path)
    centroids = read_fvecs(centroids_path)
    cluster_id = I64vecs_read(cluster_id_path)
    X = X[:, :bits]
    D = X.shape[1]
    B = D
    if B % 32 != 0:
        raise ValueError(f'RESQ projection dimension must be a multiple of 32, got {B}')
    if centroids.shape[1] != D:
        raise ValueError(
            f'Centroid dimension {centroids.shape[1]} does not match projection dimension {D}'
        )
    MAX_BD = B

    projection_path = os.path.join(path, f'RESP_C{C}_B{B}.fvecs')
    randomized_centroid_path = os.path.join(path, f'RESCentroid_C{C}_B{B}.fvecs')
    RN_path = os.path.join(path, f'RES_Rand_C{C}_B{B}.Ivecs')
    x0_path = os.path.join(path, f'RES_x0_C{C}_B{B}.fvecs')

    X_pad = np.pad(X, ((0, 0), (0, MAX_BD - D)), 'constant')
    centroids_pad = np.pad(centroids, ((0, 0), (0, MAX_BD - D)), 'constant')
    np.random.seed(0)

    # The inverse of an orthogonal matrix equals to its transpose.
    P = Orthogonal(MAX_BD)
    P = P.T

    cluster_id = np.squeeze(cluster_id)
    XP = np.dot(X_pad, P)
    CP = np.dot(centroids_pad, P)
    XP = XP - CP[cluster_id]
    bin_XP = (XP > 0)

    # The inner product between the data vector and the quantized data vector, i.e., <\bar o, o>.
    x0 = np.sum(XP[:, :B] * (2 * bin_XP[:, :B] - 1) / B ** 0.5, axis=1, keepdims=True) / np.linalg.norm(XP, axis=1,
                                                                                                        keepdims=True)

    # To remove illy defined x0
    # np.linalg.norm(XP, axis=1, keepdims=True) = 0 indicates that its estimated distance based on our method has no error.
    # Thus, it should be good to set x0 as any finite non-zero number.
    x0[~np.isfinite(x0)] = 0.8

    print(np.mean(x0))

    uint64_XP = PackBinaryCodes(bin_XP[:, :B])

    # Output
    fvecs_write(randomized_centroid_path, CP)
    I64vecs_write(RN_path, uint64_XP)
    fvecs_write(x0_path, x0)
    fvecs_write(projection_path, P)
