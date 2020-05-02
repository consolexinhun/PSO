
import random
import matplotlib.pyplot as plt
import numpy as np
NGEN = 1000  # 迭代次数
popsize=  100  # 种群个数
# var_num =  5  # 变量维度
# var_num =  10  # 变量维度
# var_num =  30  # 高维情况下需要增加种群的个数
var_num =  50  # 高维情况下需要增加种群的个数
lb = [-32.768 for _ in range(var_num)]
ub = [32.768 for _ in range(var_num)]

class PSO:
    def __init__(self):
        self.pop_x = np.zeros((popsize, var_num))
        self.pop_v = np.zeros((popsize, var_num))

        self.p_best = np.zeros((popsize, var_num))
        self.g_best = np.zeros((1, var_num))

        temp = -1 # 全局最好个体的适应度
        for i in range(popsize):
            for j in range(var_num):
                self.pop_x[i][j] = random.uniform(lb[j], ub[j])
                self.pop_v[i][j] = random.random()  # [0, 1)
            self.p_best[i] = self.pop_x[i]

            fit = self.fitness(self.p_best[i]) # 当前最好个体的适应度
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    '''
    计算适应度
    ind_var ：个体的位置
    y 函数值
    100/y 适应度， 适应度必须越大越好，并且是非负的
    '''
    def fitness(self, x):
        d = var_num
        y1 = 0.0
        for i in range(d):
            y1 += x[i]**2
        res1 = -20*np.exp(-0.2 * np.sqrt(y1/d))

        y2 = 0.0
        for i in range(d):
            y2 += np.cos(2*np.pi*x[i])
        y2 /= d
        res2 = -np.exp(y2)

        f = res1+res2
        fit = -f
        return fit 
        
    def update(self):
        c1 = 2
        c2 = 2
        w = 0.4
        for i in range(popsize):
            self.pop_v[i] = w*self.pop_v[i] + c1 * random.random()*(self.p_best[i] - self.pop_x[i]) + c2*random.random()*(self.g_best-self.pop_x[i])
            self.pop_x[i] += self.pop_v[i]

            # 防止越界
            for j in range(var_num):
                if self.pop_x[i][j] < lb[j]:
                    self.pop_x[i][j] = lb[j]
                if self.pop_x[i][j] > ub[j]:
                    self.pop_x[i][j] = ub[j]
            
            if self.fitness(self.pop_x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.pop_x[i]
            if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
                self.g_best = self.pop_x[i]
        
    def main(self):
        popobj = []
        print('-----------begin--------')
        for gen in range(NGEN):
            self.update()
            popobj.append(-self.fitness(self.g_best))

        print('位置:{}'.format(self.g_best))
        print('函数值:{}'.format(-self.fitness(self.g_best)))
        print('------end---------')
        plt.plot(popobj)
        plt.show()

if __name__ == "__main__":
    pso = PSO()
    pso.main()