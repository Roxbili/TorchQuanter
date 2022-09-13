#!/bin/bash

python3 examples/qat/qat.py --dataset "cifar10" \
    --dataset-dir "/home/LAB/leifd/dataset/cifar/cifar-10" \
    --mean 0.4914 0.4822 0.4465 --std 0.2470 0.2435 0.2616
