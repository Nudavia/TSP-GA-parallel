import copy
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from time import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

Comm = MPI.COMM_WORLD
Rank = Comm.Get_rank()
Size = Comm.Get_size()


class GA:
    def __init__(self, m, t, p_corss, p_mutate, group_count, migrate_rate, wait, filepath):
        self.filepath = filepath  # 数据源
        self.pos = []  # 坐标
        self.M = m  # 种群规模
        self.T = t  # 运行代数
        self.t = 0  # 当前代数
        self.p_corss = p_corss  # 交叉概率
        self.p_mutate = p_mutate  # 变异概率
        self.cityNum = 0  # 城市数量，染色体长度
        self.dist = []  # 距离矩阵
        self.bestDistance = -1  # 最佳长度
        self.bestPath = []  # 最佳路径
        self.oldPopulation = []  # 父代种群
        self.newPopulation = []  # 子代种群
        self.fitness = []  # 个体的适应度
        self.Pi = []  # 个体的累积概率
        self.record = []  # 记录适应度变化
        self.fitCount = 0  # 计算适应度次数
        self.groupCount = group_count  # 组数
        self.tribeCount = int(Size / group_count)  # 组内种群书
        self.migrateRate = migrate_rate  # 每个migrateRate代组间迁移一次
        self.isFinish = False  # 该种群是否不在变化
        self.wait = wait

    # 读取文件
    def readfile(self, filepath):
        infile = open(filepath)
        i = 0
        for line in infile:
            linedata = line.strip().split()
            self.pos.append([float(linedata[1]), float(linedata[2])])
            i = i + 1
            if i > 500:
                break
        infile.close()

    # 初始化dist矩阵
    def initdist(self):
        self.cityNum = len(self.pos)  # 城市数量，染色体长度
        self.dist = np.zeros([self.cityNum, self.cityNum], dtype=int)
        for i in range(self.cityNum):
            for j in range(i, self.cityNum):
                self.dist[i][j] = self.dist[j][i] = self.distance(self.pos[i], self.pos[j])

    # 随机初始化种群
    def initpopulation(self):
        # 随机初始化种群
        for k in range(self.M):
            tmp = np.arange(self.cityNum)
            np.random.shuffle(tmp)
            self.oldPopulation.append(tmp)

    # 更新个体适应度并做好记录
    def updatefitness(self):
        self.fitness.clear()
        for i in range(self.M):
            self.fitness.append(self.evaluate(self.oldPopulation[i]))
        self.record.append(np.sum(self.fitness) / self.M)

    # 计算某个染色体的实际距离作为染色体适应度
    def evaluate(self, chromosome):
        self.fitCount = self.fitCount + 1
        length = 0
        for i in range(1, self.cityNum):
            length += self.dist[chromosome[i - 1]][chromosome[i]]
        length += self.dist[chromosome[0]][chromosome[self.cityNum - 1]]  # 回到起点
        return length

    # 计算欧氏距离矩阵
    @staticmethod
    def distance(pos1, pos2):
        return np.around(np.sqrt(np.sum(np.power(np.array(pos1) - np.array(pos2), 2))))

    # 适应度转化函数
    @staticmethod
    def fitfunc(fit):
        return 10000 / fit

    # 计算种群中每个个体的累积概率
    def countrate(self):
        tmp_fit = self.fitfunc(np.array(self.fitness))
        fit_sum = np.sum(tmp_fit)
        self.Pi = tmp_fit / fit_sum
        self.Pi = list(itertools.accumulate(self.Pi))
        self.Pi[self.M - 1] = np.round(self.Pi[self.M - 1])  # 最后四舍五入保证累计概率的最后一个值为1

    # 轮盘挑选子代个体
    def selectchild(self):
        self.newPopulation.clear()
        for i in range(0, self.M):
            rate = np.random.random(1)
            for oldId in range(self.M):
                if self.Pi[oldId] >= rate:
                    self.newPopulation.append(copy.deepcopy(self.oldPopulation[oldId]))
                    break
        self.oldPopulation.clear()

    # 进化种群
    def evolution(self):
        self.selectchild()  # 选择
        rand = np.arange(self.M)
        np.random.shuffle(rand)
        for k in range(1, self.M, 2):
            rate_c = np.random.random(1)
            if rate_c < self.p_corss:  # 交叉
                self.ordercross(rand[k], rand[k - 1])
            rate_m = np.random.random(1)
            if rate_m < self.p_mutate:  # 变异
                self.variation(rand[k])
            rate_m = np.random.random(1)
            if rate_m < self.p_mutate:
                self.variation(rand[k - 1])

    # 产生2个索引，用于交叉和变异
    def randomrange(self):
        left = 0
        right = 0
        while left == right:
            left = np.random.randint(0, self.cityNum)
            right = np.random.randint(0, self.cityNum)
        ran = np.sort([left, right])
        left = ran[0]
        right = ran[1]
        return left, right

    # 变异算子，翻转一段基因
    def variation(self, k):
        ran = self.randomrange()
        left = ran[0]
        right = ran[1]
        while left < right:
            tmp = self.newPopulation[k][left]
            self.newPopulation[k][left] = self.newPopulation[k][right]
            self.newPopulation[k][right] = tmp
            left = left + 1
            right = right - 1

    # 映射交叉算子(互换片段，映射去重)
    def ordercross(self, k1, k2):
        ran = self.randomrange()
        left = ran[0]
        right = ran[1]
        map1 = {}
        map2 = {}
        old1 = copy.deepcopy(self.newPopulation[k1])
        old2 = copy.deepcopy(self.newPopulation[k2])
        for i in range(left, right + 1):
            map1[self.newPopulation[k1][i]] = self.newPopulation[k2][i]
            map2[self.newPopulation[k2][i]] = self.newPopulation[k1][i]

        for i in range(self.cityNum):
            g = self.newPopulation[k1][i]
            if i < left or i > right:
                while map2.get(g) is not None:  # 非交换部分，由于有些交换可以抵消，所以要循环映射直到找不到下一个映射
                    g = map2[g]
                self.newPopulation[k1][i] = g
            else:  # 交换的部分直接映射
                self.newPopulation[k1][i] = map1[g]

        for i in range(self.cityNum):
            g = self.newPopulation[k2][i]
            if i < left or i > right:
                while map1.get(g) is not None:
                    g = map1[g]
                self.newPopulation[k2][i] = g
            else:
                self.newPopulation[k2][i] = map2[g]

        # 只有比父母好才能替代父母
        if self.evaluate(old1) < self.evaluate(self.newPopulation[k2]):
            self.newPopulation[k2] = old1
        if self.evaluate(old2) < self.evaluate(self.newPopulation[k1]):
            self.newPopulation[k1] = old2

    # 获取最好的个体
    def getbest(self):
        best_id = np.argmin(self.fitness)
        best_distance = self.fitness[best_id]
        if self.bestDistance == -1 or best_distance < self.bestDistance:
            self.bestDistance = best_distance
            self.bestPath = copy.deepcopy(self.oldPopulation[best_id])
        else:
            self.wait = self.wait - 1

    # 开始GA
    def run(self):
        if Rank == 0:
            self.readfile(self.filepath)
        self.pos = Comm.bcast(self.pos, root=0)  # 广播坐标集
        self.initdist()  # 初始化距离矩阵
        self.initpopulation()  # 初始化种群
        self.updatefitness()  # 初始化适应度
        self.countrate()  # 初始化累计概率
        self.getbest()  # 得到最好的个体
        while self.t < self.T:
            self.t = self.t + 1
            if Rank == 0:
                print(self.t)
            self.isFinish = False
            self.evolution()
            self.oldPopulation = copy.deepcopy(self.newPopulation)
            # 组间交换
            if self.t % self.migrateRate == 0 and self.groupCount > 1:
                if Rank % self.tribeCount == 0:
                    dest = (Rank + self.tribeCount) % Size
                    src = (Rank - self.tribeCount + Size) % Size
                    Comm.send(self.oldPopulation, dest=dest)
                    self.oldPopulation = Comm.recv(source=src)
            # 组内交换
            elif self.tribeCount > 1:
                # 把老一代最好的放入现在刚成为老一代的新一代
                group = int(Rank / self.tribeCount) * self.tribeCount
                dest = (Rank + 1) % self.tribeCount + group
                src = (Rank - 1 + self.tribeCount) % self.tribeCount + group
                Comm.send(self.bestPath, dest=dest)
                self.oldPopulation[0] = Comm.recv(source=src)
            self.updatefitness()

            # 搜集适应度变化
            record = Comm.gather(self.record[self.t], root=0)  # 实际上只有0号进程获取了，对于其他进程record是None，不能直接求和除Size
            if Rank == 0:
                self.record[self.t] = np.sum(record) / Size

            self.countrate()
            self.getbest()

            # 0号判断是否完成
            if Rank == 0 and (self.record[self.t - 1] == self.record[self.t] or self.wait == 0):
                self.isFinish = True
            else:
                self.isFinish = False

            # 广播让其他进程也结束
            self.isFinish = Comm.bcast(self.isFinish, root=0)
            if self.isFinish:
                break

        # 搜集其他种群最好的个体
        best_paths = Comm.gather(self.bestPath, root=0)
        best_distances = Comm.gather(self.bestDistance, root=0)
        if Rank == 0:
            best_id = np.argmin(best_distances)
            self.bestPath = best_paths[best_id]
            self.bestDistance = best_distances[best_id]
            print("结果:")
            print("终止代数:" + str(self.t))
            print("最优路径:" + str(self.bestPath))
            print("最优距离:" + str(self.bestDistance))

    # 显示结果
    def show(self):
        plt.title('TSP-GA')
        ax1 = plt.subplot(221)
        ax1.set_title('原始坐标')
        ax1.set_xlabel('x坐标')
        ax1.set_ylabel('y坐标')
        for point in self.pos:
            plt.plot(point[0], point[1], marker='o', color='k')

        ax2 = plt.subplot(222)
        ax2.set_title('适应度变化')
        ax2.set_xlabel('代数')
        ax2.set_ylabel('平均代价')
        for i in range(1, len(self.record)):
            plt.plot([i, i - 1], [self.record[i], self.record[i - 1]], marker='o', color='k', markersize='1')

        ax3 = plt.subplot(223)
        ax3.set_title('线路')
        ax3.set_xlabel('x坐标')
        ax3.set_ylabel('y坐标')
        for point in self.pos:
            plt.plot(point[0], point[1], marker='o', color='k')
        for i in range(1, self.cityNum):
            plt.plot([self.pos[self.bestPath[i]][0], self.pos[self.bestPath[i - 1]][0]],
                     [self.pos[self.bestPath[i]][1], self.pos[self.bestPath[i - 1]][1]], color='g')
        plt.plot([self.pos[self.bestPath[0]][0], self.pos[self.bestPath[self.cityNum - 1]][0]],
                 [self.pos[self.bestPath[0]][1], self.pos[self.bestPath[self.cityNum - 1]][1]], color='g')

        plt.savefig(self.filepath + '-' + str(self.bestDistance) + '.jpg')
        plt.show()


def main():
    t1 = time()
    ga = GA(m=int(800 / Size), t=100000, p_corss=0.7, p_mutate=0.05, group_count=4, migrate_rate=100, wait=10000,
            filepath="data/pr299.txt")
    ga.run()
    t2 = time()
    if Rank == 0:
        print("耗时:" + str(t2 - t1))
        print("评估次数:" + str(ga.fitCount))
        ga.show()


if __name__ == '__main__':
    main()
