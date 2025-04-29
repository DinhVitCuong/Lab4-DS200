import os
import time
import json
import pickle
import socket
import argparse
import numpy as np
from math import ceil
from tqdm import tqdm

TCP_IP = "localhost"
TCP_PORT = 6100

def parse_args():
    p = argparse.ArgumentParser(
        description="Stream CIFAR batches over TCP to a Spark Streaming Context"
    )
    p.add_argument(
        "--folder", "-f", required=True, type=str,
        help="Path to the CIFAR folder (contains data_batch_1…5 and test_batch)"
    )
    p.add_argument(
        "--batch-size", "-b", required=True, type=int,
        help="Number of samples per batch"
    )
    p.add_argument(
        "--split", "-s", choices=("train", "test"),
        default="train", help="Which split to stream"
    )
    p.add_argument(
        "--sleep", "-t", type=float, default=3.0,
        help="Seconds to wait between batches"
    )
    p.add_argument(
        "--endless", "-e", action="store_true",
        help="Loop forever over the split"
    )
    return p.parse_args()

def get_batch_files(folder: str, split: str):
    # all CIFAR batch paths
    files = [
        os.path.join(folder, f"data_batch_{i}") for i in range(1, 6)
    ] + [os.path.join(folder, "test_batch")]
    return files[:-1] if split == "train" else [files[-1]]

def batch_generator(batch_files, batch_size):
    """
    Yields (X, y) for each batch. X is an (N,3072) numpy array,
    y is a list of length N.
    """
    for file_path in batch_files:
        with open(file_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        data = batch[b"data"]
        # ensure numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        labels = batch[b"labels"]
        n_samples = data.shape[0]
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            yield data[start:end], labels[start:end]

def stream_to_spark(args):
    batch_files = get_batch_files(args.folder, args.split)
    # each CIFAR file has 10000 samples
    batches_per_file = ceil(10000 / args.batch_size)
    total_batches = batches_per_file * len(batch_files)

    print(f"→ Streaming {args.split} split ({len(batch_files)} files), "
          f"{batches_per_file} batches each → {total_batches} total.")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((TCP_IP, TCP_PORT))
        server.listen(1)
        print(f"Waiting for Spark to connect on {TCP_IP}:{TCP_PORT}…")
        conn, addr = server.accept()
        with conn:
            print(f"Connected to {addr}!\n")
            iterator = batch_generator(batch_files, args.batch_size)
            loop_iter = iterator
            if args.endless:
                # cycle endlessly
                from itertools import cycle
                loop_iter = cycle(iterator)

            pbar = tqdm(loop_iter, total=total_batches, unit="batch")
            for X, y in pbar:
                # build one payload per batch: list of {feature-i: val, label: ...}
                payload = [
                    {**{f"feature-{i}": int(val) for i, val in enumerate(row)},
                     "label": int(lbl)}
                    for row, lbl in zip(X, y)
                ]
                data = (json.dumps(payload) + "\n").encode("utf-8")
                try:
                    conn.sendall(data)
                except BrokenPipeError:
                    pbar.write("⚠️  Connection closed by Spark. Exiting.")
                    break

                pbar.set_description(f"sent batch of {len(y)}")
                time.sleep(args.sleep)

if __name__ == "__main__":
    args = parse_args()
    stream_to_spark(args)
