import matplotlib.pyplot as plt
import numpy as np
import math as mt
import pandas as pd



class MC:
    ''' 简化考虑：有N个资产就是N期himalaya'''
    
    def __init__(self, assets, times, step):
        '''输入：       
              资产相关：1.资产的日度收益率序列 assets
              
              蒙卡相关：2.独立重复实验次数 times
                       3.单期时长 t_step（年化 一个月是1/12）
          计算：
              资产相关：1.corr  2.cov   3.mean   
              
        #收益率序列相关性质：波动率 均值 要记得年化
        n = 1000
        m = 6
        t = 2/12 #单期时长 年化表示
        d = np.random.random(m) #生成m个0-1之间的随机数作为资产方差δ2
        r = simLD(m,0.5,0) #生成半正定的相关系数矩阵            '''
          
        self.assets = assets
        self.times = times
        self.step = step
        self.n = self.assets.shape[1]
        self.corr = self.assets.corr()
        self.cov = self.assets.cov()*252 # cov = assets.cov()
        self.miu = self.assets.mean()*252
        
   
    #判断产生的corr 是否为半正定矩阵
    def is_pos_def(self,A):
        A = self.corr
        if np.array_equal(A, A.T):
            try:
                np.linalg.cholesky(A)
                return '是半正定矩阵'
            except np.linalg.LinAlgError:
                return '不是半正定矩阵-1'
        else:
            if np.allclose (A, A.T, rtol = 1e-05 , atol = 1e-08 , equal_nan = False ):
                try:
                    np.linalg.cholesky(A)
                    return '是半正定矩阵'
                except np.linalg.LinAlgError:
                    return '不是半正定矩阵-1'
            else:
                return '不是半正定矩阵-2'


    # 计算himalaya期权的payoff
    def himalaya_options_payoff(self,s_ratio_):
        thesum = 0
        s_ratio_ = self.s_ratio
        for t in range(1, self.n+1): # t = 1
            maxpos = s_ratio_[:, t-1].argmax()
            # thesum加上maxVal，然后将这一行设为 -100000，以便下期遍历不会遍历这一行了（该资产被除去）。
            thesum += s_ratio_[maxpos, t-1]
            s_ratio_[maxpos, :] = -10000
        # self.s_ratio = s_ratio_ # 该函数中的改动会影响到s_ratio变量
        return thesum


    # 蒙卡模拟
    def mc_himalaya(self):
        '''
        基于几何布朗运动模型构建的n个独立分布的资产收益率序列rt表示为：
            rt = μ + δ*et
        对符合标准正态分布的变量et进行修正得到有相关关系漂移项εt来描述资产间的耦合关系
            εt = C*et
            rt = μ + δ*C*et
        '''

        #资产年度收益率均值？
        rm = np.array(self.miu) # np.array([0]*self.n)
        #资产年度波动率
        delta2 = np.array(self.cov).diagonal()
        #Δt：单个时间步进的年化时长？[1/12]
        delta_t = np.array([self.step]*self.n)

        '''判断相关系数矩阵就是否为半正定？'''
        # judge = self.is_pos_def(self.corr)
        # print(judge)


        # 对协方差矩阵进行cholesky分解：从这里开始需要确保相关系数矩阵为半正定
        l = np.linalg.cholesky(self.cov)# 返回下三角矩阵 r = l*l.T
      
        # 存放蒙卡模拟下的资产价格净值路径 stock_p：产生times个 n*n+1的矩阵
        stock_p = np.ones((self.times,self.n,self.n+1))
        # 存放MC所有模拟次数的掉的期权价格总和
        himalaya_payoff_total = 0
        # 设定对每次循环分别固定的种子值 用于重复检查
        seed = 0
        # 开始蒙卡模拟
        for i in range(self.times):# 第i次独立重复实验
            for t in range(1,self.n+1):# 第t期
                # 生成独立的标准正态分布序列-对应当期N个资产的当期收益率
                # 设定种子值用于检查
                np.random.seed(seed)
                seed += 1
                z = np.random.randn(self.n).reshape(self.n,1)
                # 计算单次模拟中的资产净值路径
                stock_p[i,:,t] = stock_p[i,:,t-1] * np.squeeze(np.array(
                                                              np.exp((np.matrix(rm - delta2 / 2) * delta_t[t - 1] ).reshape(self.n, 1) \
                                                                      + np.matmul(l, z) * np.sqrt(delta_t[t - 1]))))
            s_diff = stock_p[i, :, 1:] - stock_p[i, :, :-1]
            # 计算单次模拟中各资产各期收益率
            self.s_ratio = s_diff / stock_p[i, :, :-1]
            # 计算Himalaya期权payoff
            himalaya_payoff_total += self.himalaya_options_payoff(self.s_ratio)
            
        self.himalaya_payoff_mc = himalaya_payoff_total/self.times
        
        # 无风险利率 rf  折现payoff到期初
        rf = 0
        self.himalaya_price = self.himalaya_payoff_mc * np.exp(-1*rf*self.n*self.step)
        return self.himalaya_price


#%%

datapath = 'C:\\Users\\Sherry\\Desktop\\Files\\实习\\广发\\奇异期权定价\\代码\\data.xlsx'
data = pd.read_excel(datapath)
data = data.iloc[:,1:]

assets = data
times = 100
step = 1/12
ddd = MC(assets, times, step)
price = ddd.mc_himalaya()






















