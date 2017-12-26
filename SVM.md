---
title: SVM
tags: 向量机
grammar_cjkRuby: true
---

### 决策边界：
		“越胖越好”，泛化能力越强
![图转换][1]

	  将求决策边界转化为求点到平面的距离，可使用法向量等求解

![举例图][2]
	  
	  yi为真实值，y(xi)为预测值
	  目的：找到一个决策边界将两类数据完全分离，但是为了有更好的鲁棒性和泛化能力，有时并不时将所有给出的数据分开就是最好的
	  转换为：找到一条直线，使得立直线最近的点能够最远

![数学表达][3]

	 	先找出最小的点，在使得这个最小的点的距离最大

![数学转换][4]

		转换时添加系数1/2是为了后面的求偏导化简
		拉格朗日乘子法中s.t为约束条件（即将前面yi（……）>=1转换为左边与右边相减）

![拉格朗日转换][5]

		转化为求解α
		再次转换为对偶问题，求极小值

![求解偏导][6]


![求解1][7]


![求解2][8]
	
		拉格朗日乘子法中默认的条件αi>=0
![举例][9]

![举例求解][10]


![继续求解][11]
	
		使用α1和α2替换α3
		求解的α为负值时不满足要求
![求解权值和偏向][12]

		因为求解得到的α2=0，对应的点为x2，α2=0表示求解和构建决策平面时不需要考虑x2，只需要考虑最近的x1和x3就可以
### 软间隔问题：
![问题][13]


![解决方法][14]

	C为权重系数

![求解][15]


  ### 向量核变换：
  
 	低维不可分问题可映射到高维解决（足够高的维度可以解决更复杂的问题）
![核计算][16]

	K<x,y>=(<x,y>)^2
	先平方后计算内积=先计算内积再进行平方
	使用核计算，结果一致，但是计算量大大减小，速度大大提升

![高斯核函数][17]

	使用核函数对原始空间进行转换，由低维转换到高维


### SMO算法（序列最小化）：
	现在的SVM基本使用SMO算法求解。
![SMO思想][18]

![思想过程][19]
	
	求解α1，α2时把其他的α固定，看成常数
	
![过程续][20]

![求解过程][21]

	先对α1求偏导并令其为0，可以解得α2的值，回带求得α1
	记求导式为式1，f(x)为式2，Ei为式3，则将式1带入式2，可得K式，再将式3带入K式即可得到最后两式的化简结果（次化简计算量很大，化简难），得到优化前后新旧值之间的关系
	
![约束][22]
	
	yi值只能为1或者-1，所以α值有了界限
	图中L为下界，H为上界
	求出的α要满足约束条件

#### 简化版SMO算法的代码实现：

	def clipAlpha*aj,H,L):
			if aj>H:
				aj=	H
			if L>aj:
				aj=L
			return aj
			
	def selectJrand(i,m):
		j=i
		while(j==i):
				j=int(np.random.uniform(0,m))
		return j
	
	//参数依次为特征值X，Y值，C值（权重系数），边界容忍值(容忍程度)，最大迭代次数
	def smosimple(dateMatin,classLables,C,toler,maxIter):
			//初始化
			dataMatrix=np.mat(dateMatin)
			labelMat=np.mat(classLabels).transpose
			b=0
			m,n=np.shape(dataMatrix)  //m表示数据样本中的个数
			alphas=np.mat(np.zeros(m,1))
			
			iter=0
			while(iter<maxIter):
					alphaPairsChangeed=0
					for i in range(m): //每次选两个α值
							fxi=float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
							Ei=fxi-float(labelMat[i])
							if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and 																				(alphas[i]>0))：
									j=selectJrand(i,m)
									fxj=float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
									Ej=fxj-float(labelMat[j])
									alphaiold=alphas[i].copy()
									alphajold=alphas[j].copy()
									if (labelMat[i]!=labelMat[j]):	//根据约束条件计算上下界
											L=max(0,alphas[j]-alphas[i])
											H=min(C,C*alphas[j]-alphas[i])
									else
											L=max(0,alphas[j]+alphas[i]-C)
											H=min(C,C*alphas[j]+alphas[i])
									
									if L==H:
											print("L==H")
											continue
									eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-															dataMatrix[j,:]*dataMatrix[j,:].T
									if eta >= 0:
											print("μ===")
											continue
									alphas[j]-=labelMat[j]*(Ei-Ej)/eta
									alphas[j]=clipApha(alphas[j],H,L)	//由上下界确定alphas的值是否满足要求
									
									if (abs(alphas[j]-alpajold)<0.00001):
											print("==============")
											continue
									alphas[i]+= labelMat[j]*labelMat[i]*(alphajold-alphas[j])
									
									 b1=b-Ei-labelMat[i]*(alphas[i]-alphaiold)*dataMatrix[i,:]*dataMatrix[i,:].T-																					labelMat[i]
									 b2=b-Ej-labelMat[i]*(alphas[i]-alphaiold)*dataMatrix[i,:]*dataMatrix[j,:].T-																					labelMat[j]
									if (a<alphas[i]) and (C>alphas[i]):
											b=b1
									elif (0<alphas[j]) and (C>alphas[j]):
											b=b2
									else:
											b=(b1+b2)/2.0
									alphaPairsChanged+=1
									print("--%d---%d---%d---" *% (iteri,i,alphaPairsChanged))
					if (alphaPairsChanged==0):
							iter += 1
					else:
							iter = 0
					print("Iter=%d" % iter)
					
			return b,alphas	
									
									
											
  [1]: ./images/1514171734157.jpg
  [2]: ./images/1514171880053.jpg
  [3]: ./images/1514172206185.jpg
  [4]: ./images/1514172437904.jpg
  [5]: ./images/1514172692100.jpg
  [6]: ./images/1514172826410.jpg
  [7]: ./images/1514172978952.jpg
  [8]: ./images/1514173081280.jpg
  [9]: ./images/1514174131634.jpg
  [10]: ./images/1514173337773.jpg
  [11]: ./images/1514173557556.jpg
  [12]: ./images/1514174161033.jpg
  [13]: ./images/1514174522754.jpg
  [14]: ./images/1514174693374.jpg
  [15]: ./images/1514175176876.jpg
  [16]: ./images/1514187529601.jpg
  [17]: ./images/1514187949947.jpg
  [18]: ./images/1514188085877.jpg
  [19]: ./images/1514188308740.jpg
  [20]: ./images/1514188493791.jpg
  [21]: ./images/1514188617580.jpg
  [22]: ./images/1514189669589.jpg