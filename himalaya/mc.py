import numpy as np
import pandas as pd
import cvxpy as cvx


class MC:
    ''' 简化考虑：有N个资产就是N期himalaya
        Himalaya payoff:在每个收益表现冻结时点挑选出表现最好的资产，记录收益率，并剔除该资产
                        直至将所有资产剔除，加总所有记录的收益率，并除以期数？该值作为Himalaya的收益率
                        
                        该期权好在：便宜（如果每期丢掉的是表现最差的资产，该期权价格还要高点）
                             差在：表现最差的资产持有时间最长，拉低平均收益率'''
    
    def __init__(self, assets, times, step, rf, mineig,
                 use_give = False,
                 type_ = 'GF1'):
        '''输入：       
              资产相关：1.资产的日度收益率序列 assets  此时use_give = False 默认为False
                         /或者是给定scenario下的[corr,sigma]  此时use_give = True
    
              
              蒙卡相关：2.独立重复实验次数 times
                       3.单期时长 t_step（年化 一个月是1/12）
                       
              其他：4.使用payoff计算蒙卡price时所用的无风险利率rf
                    5.如果资产corr矩阵为非半正定那么对corr进行修改设定的修改后corr矩阵最小特征值mineig
                    6.是否使用给定scenario下的corr和cov数据  True为使用 则在原来的asset参数中输入给定数据[corr,sigma]
                                                          False为不使用 则还是使用asset的data计算corr与cov
                    7.type:
                            ##origin:SUM(节点的累计收益率)/N期
                            
                            GF1：MAX(SUM(节点的累计收益率),0)
                            GF2: MAX(SUM(节点的区间收益率),0)
                            LF1: SUM(max(节点的累计收益率),0)
                            LF2: SUM(max(节点的区间收益率),0)
                        
                        
            输出：mc模拟的喜马拉雅期权价格'''

        if use_give is True:
            self.corr = assets[0]
            self.delta2 = assets[1] # 年化
            self.cov = np.diag(self.delta2)**(1/2) @ self.corr @ np.diag(self.delta2)**(1/2)
            self.n = self.corr.shape[1]
            self.miu = [0] * self.n
        else:
            self.assets = assets            
            self.n = self.assets.shape[1]
            self.corr = self.assets.corr()
            self.cov = self.assets.cov()*252 # cov = assets.cov()
            self.delta2 = np.array(self.cov).diagonal()
            self.miu = self.assets.mean()*252
        
        self.times = times
        self.step = step
        self.rf = rf
        self.mineig = mineig
        self.type = type_

    #判断产生的corr 是否为半正定矩阵
    def is_pos_def(self):
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

    # 通过cvx求解一个距离给定相关矩阵最近的&符合条件的corr矩阵：
    # 保证求出来的矩阵1.半正定 2.对角线为1
    def trans_PD(self):#m 为设定的最小特征值 0.01
    
        origin = self.corr
        n = self.n
        m = self.m
        x = cvx.Variable((n,n))#,symmetric=True) 
        objective = cvx.Minimize(cvx.norm(x - origin,'fro')) 
        constraints = [cvx.PSD(x-m*np.eye(n,n)),cvx.diag(x)==1,x==x.T] 
        prob = cvx.Problem(objective, constraints) 
        prob.solve() 
        change_rrho = x.value
        
        # print(self.is_pos_def(change_rrho))
        
        return change_rrho

    # 计算himalaya期权的payoff
    # 假设在节点的确收到、给出现金流
    def himalaya_options_payoff(self,p):
        
        thesum = 0
        s_ratio_copy = self.s_ratio[p,:,:].copy()
        for t in range(1, self.n+1): # t = 1
            # 找出表现最好的资产位置 
            # 传入s_ratio代表区间收益率则就是区间收益率最大 传入累计收益率就是累计收益率最大
            # s_ratio n*n
            maxpos = s_ratio_copy[:, t-1].argmax()
            # 根据不同的payoff结构计算Himalaya期权的payoff（有贴现因子）
            if self.type == 'GF1' or self.type == 'GF2':# 1：节点累计收益率/2：节点区间收益率
                thesum += s_ratio_copy[maxpos, t-1] * np.exp(self.rf*(self.n - t)*self.step) # 先全部贴现到期末 最后全部折现回初期t0
                # 将需要被剔除的资产这一行（所有期的收益率数据）设为 -100000
                s_ratio_copy[maxpos, :] = -10000            
                # 期权 可以不行权？则收益率就是0
                result = max(thesum,0)
            elif self.type == 'LF1' or self.type == 'LF2':# 1：节点累计收益率/2：节点区间收益率
                thesum += max(s_ratio_copy[maxpos, t-1] * np.exp(self.rf*(self.n - t)*self.step),0) # 先全部贴现到期末 最后全部折现回初期t0
                s_ratio_copy[maxpos, :] = -10000            
                # 期权 可以不行权？则收益率就是0
                result = thesum
                

        return result 


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
        delta2 = np.array(self.delta2)
        #Δt：单个时间步进的年化时长？[1/12]
        delta_t = np.array([self.step]*self.n)

        '''判断相关系数矩阵是否为半正定？'''
        judge = self.is_pos_def()
        if judge == '是半正定矩阵':
            pass
        elif judge == '不是半正定矩阵-1':
            print('调整corr矩阵至半正定')
            # 使用cxvpy计算修改corr至半正定 再计算cov
            self.corr = self.trans_PD()
            self.cov = np.diag(self.delta2)**(1/2) @ self.corr @ np.diag(self.delta2)**(1/2)

        # 对协方差矩阵进行cholesky分解：从这里开始需要确保协方差矩阵为半正定
        l = np.linalg.cholesky(self.cov)# 返回下三角矩阵 r = l*l.T
      
        # 存放蒙卡模拟下的资产价格净值路径 stock_p：产生times个 n*n+1的矩阵
        self.stock_p = np.ones((self.times,self.n,self.n+1))
        # 存放每期模拟的收益率路径 
        self.s_ratio = np.zeros((self.times,self.n,self.n))
        # 存放MC所有模拟s_ratio
        himalaya_payoff_total = 0

        # 开始蒙卡模拟
        for i in range(self.times):# 第i次独立重复实验
            for t in range(1,self.n+1):# 第t期
                #生成n*1随机数序列
                z = np.random.randn(self.n).reshape(self.n,1)

                # 计算单次模拟中的资产净值路径
                self.stock_p[i,:,t] = self.stock_p[i,:,t-1] * \
                                            np.squeeze(np.array(
                                            np.exp((np.matrix(rm - delta2 / 2) * delta_t[t - 1] ).reshape(self.n, 1) \
                                                    + np.matmul(l, z) * np.sqrt(delta_t[t - 1]))
                                                                )
                                                        )
            # 计算单次模拟中各资产各期收益率
            if self.type == 'GF1' or self.type == 'LF1':  # 累计收益率
                s_diff = self.stock_p[i, :, 1:] - np.ones((self.n,self.n)) # self.stock_p[i, :, :-1]
                self.s_ratio[i,:,:] = s_diff

            elif self.type == 'GF2' or self.type == 'LF2': # 单期收益率
                s_diff = self.stock_p[i, :, 1:] - self.stock_p[i, :, :-1]
                self.s_ratio[i,:,:] = s_diff / self.stock_p[i, :, :-1]

            # 计算Himalaya期权payoff
            himalaya_payoff_total += self.himalaya_options_payoff(i)
            
        self.himalaya_payoff_mc = himalaya_payoff_total/self.times
        
        '''无风险利率 rf  折现payoff到期初 设名义本金为1 Himalaya期权价格=（名义本金*payoff）*折现'''
        self.nominal = 1
        self.himalaya_price = (self.nominal * self.himalaya_payoff_mc) * np.exp(-1*self.rf*self.n*self.step)
        
        return self.himalaya_price



















