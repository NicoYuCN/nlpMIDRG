from step2 import run
from step3 import Result
from step4 import Scorer
from step6 import SimCTGGPT
from step7 import Result2

import pandas as pd
import numpy as np
import random
import torch


r = [0.6]

LS = []

for kk in r:

    ls = []

    """模型训练"""
    model = run(kk=kk)
    net = SimCTGGPT(model.gpt)

    """对比搜索解码"""
    ref, gt = Result2(net, "cs", beam_width=5, number=2)
    print("多样性对比搜索")
    scorer4 = Scorer(ref, gt)
    temp = scorer4.compute_scores()
    ls.append(temp)

    ref, gt = Result2(net, "gs", beam_width=5, number=1)
    print("贪婪搜索")
    scorer4 = Scorer(ref, gt)
    temp = scorer4.compute_scores()
    ls.append(temp)

    """对比搜索解码"""
    ref, gt = Result2(net, "cs", beam_width=5, number=1)
    print("对比搜索")
    scorer4 = Scorer(ref, gt)
    temp = scorer4.compute_scores()
    ls.append(temp)

    ref, gt = Result2(net, "ns", beam_width=5, number=1)
    print("ns搜索")
    scorer4 = Scorer(ref, gt)
    temp = scorer4.compute_scores()
    ls.append(temp)

    ref, gt = Result2(net, "topk", beam_width=5, number=1)
    print("topk搜索")
    scorer4 = Scorer(ref, gt)
    temp = scorer4.compute_scores()
    ls.append(temp)

    print(ls)

    LS.append(ls)


for i in LS:
    print(i)


# """模型训练"""
# model = run(kk=1)
# net = SimCTGGPT(model.gpt)
#
# """对比搜索解码"""
# ref, gt = Result2(net, "cs", beam_width=5, number=1)
# print("对比搜索")
# scorer4 = Scorer(ref, gt)
# temp = scorer4.compute_scores()
# ls.append(temp)
#
# """对比搜索解码"""
# ref, gt = Result2(net, "cs", beam_width=5, number=2)
# print("对比搜索")
# scorer4 = Scorer(ref, gt)
# temp = scorer4.compute_scores()
# ls.append(temp)

# print(ls)


# """对比搜索解码"""
# ref, gt = Result2(net, "cs", beam_width=10, number=1)
# print("对比搜索")
# scorer4 = Scorer(ref, gt)
# temp = scorer4.compute_scores()
# ls.append(temp)
#
# """对比搜索解码"""
# ref, gt = Result2(net, "cs", beam_width=5, number=4)
# print("对比搜索")
# scorer4 = Scorer(ref, gt)
# temp = scorer4.compute_scores()
# ls.append(temp)
#
# """对比搜索解码"""
# ref, gt = Result2(net, "cs", beam_width=10, number=2)
# print("对比搜索")
# scorer4 = Scorer(ref, gt)
# temp = scorer4.compute_scores()
# ls.append(temp)
#
# """对比搜索解码"""
# ref, gt = Result2(net, "cs", beam_width=10, number=3)
# print("对比搜索")
# scorer4 = Scorer(ref, gt)
# temp = scorer4.compute_scores()
# ls.append(temp)
#
# """对比搜索解码"""
# ref, gt = Result2(net, "cs", beam_width=10, number=4)
# print("对比搜索")
# scorer4 = Scorer(ref, gt)
# temp = scorer4.compute_scores()
# ls.append(temp)

#
# """贪婪搜索解码"""
# ref, gt = Result2(net, "gs")
# print("贪婪搜索")
# scorer1 = Scorer(ref, gt)
# temp = scorer1.compute_scores()
# ls.append(temp)
#
# """集束搜索解码"""
# ref, gt = Result2(net, "bs")
# print("集束搜索")
# scorer2 = Scorer(ref, gt)
# temp = scorer2.compute_scores()
# ls.append(temp)
#
# """top-k解码"""
# ref, gt = Result2(net, "tpk")
# print("topk搜索")
# scorer3 = Scorer(ref, gt)
# temp = scorer3.compute_scores()
# ls.append(temp)
#
# """top-p解码"""
# ref, gt = Result2(net, "ns")
# print("topp搜索")
# scorer3 = Scorer(ref, gt)
# temp = scorer3.compute_scores()
# ls.append(temp)

# print(ls)
