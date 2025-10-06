import os
import numpy as np
import pandas as pd
import dolphindb as ddb
from src.FactorEva import OptimizeFactorBackTest
import riskfolio as rp

def myOptimizeFunc(self: OptimizeFactorBackTest) -> pd.DataFrame:
    """
    组合优化回测回调函数
    solve Algorithm:
    min 0.5xTHx + fTx
    cons. Ax≤b + Aeqx=beq
    f是一个因子, H是多个因子计算的结果
    self.optData: symbol TradeDate factor1 factor2 ... factorN

    return symbol TradeDate TradeTime weightName weight
    """
    weightName = "testWeight"
    # 数据
    data = self.optData
    sliceData = data[data["TradeDate"]==data["TradeDate"].max()].reset_index(drop=True)
    # 收益模型
    returnCol = self.retFactor  # 预期收益率因子名称
    # 风险模型
    factorCols = [i for i in data.columns if i not in ["symbol","TradeDate","TradeTime",returnCol]] # 这里保险一点,万一没有那个因子的数据就会报错
    # 股票池选择 -> 选择预期收益率TOPK%股票
    k = 30
    # symbolList = sliceData[sliceData[returnCol]>=np.percentile(sliceData[returnCol]),k]["symbol"]
    symbolList = sliceData[sliceData[returnCol].rank(ascending=True)<=k]["symbol"].tolist()
    # 等权投资
    result = pd.DataFrame({
        "symbol": symbolList,
        "TradeDate": [self.currentDate]*len(symbolList),
        "TradeTime": [pd.Timestamp("1970-01-01 15:00:00")]*len(symbolList),
        "weightName": [weightName]*len(symbolList),
        "weight": [1/len(symbolList)]*len(symbolList)
    })
    return result
