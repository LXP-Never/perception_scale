# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/3/20
"""

"""
import numpy as np

mag = np.random.randn(64, 32)
bank = np.random.randn(16, 32)
# print("mag是否可逆",np.linalg.det(mag))
# print("bank是否可逆",np.linalg.det(bank))

# mel = np.dot(mag, bank.T)  # (64, 16)
bank_1 = np.linalg.inv(bank.T)
mel = np.dot(mag, bank_1)  # (64, 16)
print("mel", mel.shape)

mag_T = np.dot(mel, bank)
print("mag_T", mag_T.shape)  # (64, 32)

print("核对一下数据")
print("mag", mag[0][:10])
print("mag_T", mag_T[0][:10])



