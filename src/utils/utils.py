import cv2
# import pandas as pd
# import joblib
# import sklearn.linear_model
from numpy import *
# from sklearn import ensemble
# from sklearn.cluster import KMeans
# from sklearn.metrics import plot_roc_curve
# from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import Lasso
# from sklearn.svm import SVR, SVC
from scipy.stats import pearsonr
# from pandas.core.frame import DataFrame
# import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeRegressor


def saveModel(pkl_filename, obj):
    '''
    持久化对象

    Parameters
    ----------
        pkl_filename : string
            需要保存的对象的路径。默认保存位置在当前文件夹下，要保存到其他地方，需要把路径写完整。
            文件类型为.pkl

        obj : object
            需要保存的对象

    '''
    joblib.dump(filename=pkl_filename, value=obj)


def loadModel(pkl_filename):
    '''
    读取对象

    Parameters
    ----------
        pkl_filename : 
            需要读取的对象的路径。默认读取位置在当前文件夹下，要读取其他地方的对象，需要把路径写完整。
            文件类型为.pkl

    Returns
    -------
        obj : object
            需要读取的对象

    '''
    obj = joblib.load(pkl_filename)
    return obj


def findBestParmByGridSearchCv(X, y, estimator, parm_grid, cv_num=5, n_jobs=-1):
    '''
    网格搜索寻找最优参数
    
    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]
        
        estimator :数据类型为'object'
            基学习器对象。
            该对象被认为是实现了sklearn estimator接口的对象。按需求直接使用sklearn中的类对象。
            例如需要使用knn回归来作为基学习器，就必须要导入KNeighborsRegressor这个类，并且将该类的对象赋值给estimator参数,base_estimator=KNeighborsRegressor()。
            具体学习器对象请看下面Notes
        
        parm_grid : 数据类型为'字典'
            待优化参数
            key为需要优化的参数名。value为列表。
            例如parm_grid = {"C": [0.1, 1.0], "gamma": [0.1], "epsilon": [0.1, 1.0]}
            
        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 5
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值
        
        n_jobs : 数据类型为'int'
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 数据类型为'字典'
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'n_estimators': 86}

        best_score : 数据类型为'float'
            模型交叉验证得分
        
    Examples
    --------
        parm_grid = {'n_neighbors': np.arange(n_neighbors_start, n_neighbors_end,n_neighbors_step)}
        best_parm, best_score = findBestParmByGridSearchCv(X, y, estimator=KNeighborsRegressor(), parm_grid=parm_grid)  

    Notes
    --------
        knn作为基学习器，base_estimator = KNeighborsRegressor()
        决策树作为基学习器，base_estimator = DecisionTreeRegressor()
        随机森林作为基学习器，base_estimator = RandomForestRegressor()
        svm作为基学习器，base_estimator = SVR()    

    '''
    grid = GridSearchCV(estimator=estimator, param_grid=parm_grid, cv=cv_num, n_jobs=n_jobs).fit(X, y)
    return grid.best_params_, grid.best_score_


def findBestParmByRandomizedSearchCV(X, y, estimator, parm_list, cv_num, n_iter=20, n_jobs=-1):
    '''
    随机搜索寻找较优参数
    
    Parameters
    ----------
        X_train : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y_train : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]
        
        estimator :数据类型为'object'
            基学习器对象。
            该对象被认为是实现了sklearn estimator接口的对象。按需求直接使用sklearn中的类对象。
            例如需要使用knn回归来作为基学习器，就必须要导入KNeighborsRegressor这个类，并且将该类的对象赋值给estimator参数,base_estimator=KNeighborsRegressor()。
            具体学习器对象请看下面Notes
    
        parm_list : 数据类型为'字典'
            待优化参数
            key为需要优化的参数名。value为列表。
            例如parm_grid = {"C": [0.1, 1.0], "gamma": [0.1], "epsilon": [0.1, 1.0]}
    
        n_iter : 数据类型为'int'
            训练次数，次数越大精度越高。默认n_iter=20

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 5
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值
        
        n_jobs : 数据类型为'int'
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 数据类型为'字典'
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'n_estimators': 86}

        best_score : 数据类型为'float'
            模型交叉验证得分

    Examples
    --------
        parm_grid = {'n_neighbors': np.arange(n_neighbors_start, n_neighbors_end,n_neighbors_step)}
        best_parm, best_score = findBestParmByRandomizedSearchCV(X, y, estimator=KNeighborsRegressor(), parm_grid=parm_grid, cv_num=10, n_iter=20)  

    Notes
    --------
        knn作为基学习器，base_estimator = KNeighborsRegressor()
        决策树作为基学习器，base_estimator = DecisionTreeRegressor()
        随机森林作为基学习器，base_estimator = RandomForestRegressor()
        svm作为基学习器，base_estimator = SVR()  

    '''
    rd = RandomizedSearchCV(estimator=estimator, param_distributions=parm_list, n_iter=n_iter, n_jobs=n_jobs,
                            cv=cv_num).fit(X, y)
    return rd.best_params_, rd.best_score_


def svrParm(X, y, C, gama, epsilon, cv_num=3, n_iter=20, n_jobs=-1):
    '''
    该方法为SVM回归模型中的三个超参数自动调参。这两个超参数分别是惩罚系数C，RBF核函数的系数γ和损失距离度量ϵ

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        C : 数据类型为'类数组'(list or numpy.narray)
            惩罚系数C，对误差的容忍度。
            可以给定一个确定的量，例如C = [1.0]。
            如果不确定的话，也可以给定一个'类数组'，其中包含多个可能的取值，程序会自动选取最优值。例如C = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]。
            可以从10的幂次序列开始训练，根据效果微调。c越高，越容易过拟合。过拟合会使得训练样本的准确率高，测试样本准确率低。

        gama : 数据类型为'类数组'(list or numpy.narray)
            超参数gama。
            可以给定一个确定的量，例如gama = [0.1]。
            如果不确定的话，也可以给定一个'类数组'，其中包含多个可能的取值，程序会自动选取最优值。例如gama = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]。
            可以从10的幂次序列开始训练，根据效果微调。gama越高，越容易过拟合。过拟合会使得训练样本的准确率高，测试样本准确率低。

        epsilon : 数据类型为'类数组'(list or numpy.narray)
            超参数epsilon。
            可以给定一个确定的量，例如epsilon = [0.1]。
            如果不确定的话，也可以给定一个'类数组'，其中包含多个可能的取值，程序会自动选取最优值。例如epsilon = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]。
            可以从10的幂次序列开始训练，根据效果微调。epsilon越小，越容易过拟合。过拟合会使得训练样本的准确率高，测试样本准确率低。

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_iter : 数据类型为'int'
            训练次数，次数越大精度越高。默认n_iter=20

        n_jobs : 数据类型为'int'
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 数据类型为'字典'
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'C': 10.0, 'epsilon': 0.0, 'gamma': 0.7142857142857143}

        best_score : 数据类型为'float'
            模型交叉验证得分

    Examples
    --------
        best_parm, best_score = svrParm(X_train, y_train, C=C, gama=gama, epsilon=epsilon, cv_num=cv_num, n_iter=n_iter)

    '''
    # {'C': 10.0, 'epsilon': 0.0, 'gamma': 0.7142857142857143}
    parm_list = {"C": C, "gamma": gama, "epsilon": epsilon}
    best_params, best_score = findBestParmByRandomizedSearchCV(X, y, SVR(), cv_num=cv_num, parm_list=parm_list,
                                                               n_iter=n_iter, n_jobs=n_jobs)
    # best_params, best_score = findBestParmByGridSearchCv(X, y, SVR(), parm_list)
    return best_params, best_score


def svcParm(X, y, C=None, gama=None, cv_num=3, n_iter=20, n_jobs=-1):
    '''
    该方法为SVM分类模型中的两个超参数自动调参。这两个超参数分别是惩罚系数C和RBF核函数的系数γ

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        C : 数据类型为'类数组'(list or numpy.narray)
            惩罚系数C，对误差的容忍度。
            可以给定一个确定的量，例如C = [1.0]。
            如果不确定的话，也可以给定一个'类数组'，其中包含多个可能的取值，程序会自动选取最优值。例如C = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]。
            可以从10的幂次序列开始训练，根据效果微调。c越高，越容易过拟合。过拟合会使得训练样本的准确率高，测试样本准确率低。

        gama : 数据类型为'类数组'(list or numpy.narray)
            超参数gama。
            可以给定一个确定的量，例如gama = [0.1]。
            如果不确定的话，也可以给定一个'类数组'，其中包含多个可能的取值，程序会自动选取最优值。例如gama = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]。
            可以从10的幂次序列开始训练，根据效果微调。gama越高，越容易过拟合。过拟合会使得训练样本的准确率高，测试样本准确率低。

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_iter : 数据类型为'int'
            训练次数，次数越大精度越高。默认n_iter=20

        n_jobs : 数据类型为'int'
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 数据类型为'字典'
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'C': 10.0, 'gamma': 0.7142857142857143}

        best_score : 数据类型为'float'
            模型交叉验证得分

    '''
    parm_list = {"C": C, "gamma": gama}
    best_params, best_score = findBestParmByRandomizedSearchCV(X, y, SVC(), parm_list=parm_list, cv_num=cv_num,
                                                               n_iter=n_iter, n_jobs=n_jobs)
    # grid = GridSearchCV(SVC(), param_grid={"C": C, "gamma": gama}, cv=cv_num)
    return best_params, best_score


def rfRegressionParm(X, y,
                     n_tree_start, n_tree_end, n_tree_step,
                     max_depth_start, max_depth_end, max_depth_step,
                     min_samples_leaf_start, min_samples_leaf_end, min_samples_leaf_step,
                     cv_num=3, n_iter=20, n_jobs=-1):
    '''
    优化随机森林回归的中决策树的个数

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        n_tree_start : 数据类型为'int'
            随机森林中决策树的数量。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取较优参数。
            例如n_tree_start = 1

        n_tree_end : 数据类型为'int'
            随机森林中决策树的数量。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取较优参数。
            例如n_tree_start = 200

        n_tree_step : 数据类型为'int'
            每次测试时的步长。默认n_tree_step = 10

        max_depth_start : 数据类型为'int'
            随机森林中树的最大深度。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取较优参数。
            例如max_depth_start = 1

        max_depth_end : 数据类型为'int'
            随机森林中树的最大深度。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取较优参数。
            例如max_depth_start = 50

        max_depth_step : 数据类型为'int'
            每次测试时的步长。默认max_depth_step = 1
            例如当步长max_depth_step = 3时，左端点max_depth_start = 1，右端点max_depth_end = 10，此时树的最大深度就在[1, 4, 7]中取一个较优值。

        min_samples_leaf_start : 数据类型为'int'
            叶子节点最少的样本数。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取较优参数。
            例如min_samples_leaf_start = 1。

        min_samples_leaf_end : 数据类型为'int'
            叶子节点最少的样本数。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取较优参数。
            例如min_samples_leaf_start = 100

        min_samples_leaf_step : 数据类型为'int'
            每次测试时的步长。默认min_samples_leaf_step = 1
            例如当步长min_samples_leaf_step = 3时，左端点min_samples_leaf_step = 1，右端点min_samples_leaf_step = 10，此时叶子节点最少的样本数就在[1, 4, 7]中取一个较优值。

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_iter : 数据类型为'int'
            训练次数，次数越大精度越高。默认n_iter=20

        n_jobs : 数据类型为'int'
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 字典
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'n_estimators': 191, 'min_samples_leaf': 3, 'max_features': 'log2', 'max_depth': 16}

        best_score : float
            模型交叉验证得分

    Examples
    --------
        best_parm, best_score = rfRegressionParm(X_train, y_train,
                                                1, 200, 2,
                                                1, 50, 1,
                                                1, 20, 1,
                                                10, 20)

    '''
    parm_grid = {'n_estimators': np.arange(n_tree_start, n_tree_end, n_tree_step),
                 'max_depth': np.arange(max_depth_start, max_depth_end, max_depth_step),
                 'max_features': ['auto', 'sqrt', 'log2'],
                 'min_samples_leaf': np.arange(min_samples_leaf_start, min_samples_leaf_end, min_samples_leaf_step)
                 }
    best_parm, best_score = findBestParmByRandomizedSearchCV(X, y, estimator=ensemble.RandomForestRegressor(),
                                                             parm_list=parm_grid, cv_num=cv_num, n_iter=n_iter,
                                                             n_jobs=n_jobs)
    return best_parm, best_score


def dtRegressionParm(X, y, max_depth_start, max_depth_end, max_depth_step, cv_num=3, n_iter=20, n_jobs=-1):
    '''
    优化CART回归中树的最大深度

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        max_depth_start : 数据类型为'int'
            树的最大深度。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            例如max_depth_start = 1

        max_depth_end : 数据类型为'int'
            树的最大深度。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取最优参数。
            例如max_depth_end = 50

        max_depth_step : 数据类型为'int'
            每次测试时的步长。默认max_depth_step = 1
            例如当步长max_depth_step = 3时，左端点max_depth_start = 1，右端点max_depth_end = 10，此时树的最大深度就在[1, 4, 7]中取一个最优值

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_iter : 数据类型为'int'
            训练次数，次数越大精度越高。默认n_iter=20

        n_jobs : 数据类型为'int'
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 字典
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'max_depth': 16}

        best_score : float
            模型交叉验证得分

    '''
    parm_list = {'max_depth': np.arange(max_depth_start, max_depth_end, max_depth_step)}
    best_parm, best_score = findBestParmByRandomizedSearchCV(X, y, estimator=DecisionTreeRegressor(),
                                                             parm_list=parm_list, cv_num=cv_num, n_iter=n_iter,
                                                             n_jobs=n_jobs)
    return best_parm, best_score


def knnRegressionParm(X, y, n_neighbors_start, n_neighbors_end, n_neighbors_step, cv_num, n_jobs):
    '''
    优化knn回归中的K值
    
    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        n_neighbors_start : 数据类型为'int'
            K值。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            例如n_neighbors_start = 1

        n_neighbors_end : 数据类型为'int'
            K值。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取最优参数。
            例如n_neighbors_end = 50

        n_neighbors_step : 数据类型为'int'
            每次测试时的步长。默认n_neighbors_step = 1
            例如当步长n_neighbors_step = 3时，左端点n_neighbors_start = 1，右端点n_neighbors_end = 10，此时K值就在[1, 4, 7]中取一个较优值。

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_jobs : 数据类型为'int'
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu
        
    Returns
    -------
        best_parm : 字典
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'n_neighbors': 5}

        best_score : float
            模型交叉验证得分

    '''
    parm_grid = {'n_neighbors': np.arange(n_neighbors_start, n_neighbors_end, n_neighbors_step)}
    best_parm, best_score = findBestParmByGridSearchCv(X, y, estimator=KNeighborsRegressor(), parm_grid=parm_grid,
                                                       cv_num=cv_num, n_jobs=n_jobs)
    return best_parm, best_score


def baggingRegressionParm(X, y, base_estimator, n_estimators_start, n_estimators_end, n_estimators_step, cv_num=3,
                          n_jobs=-1):
    '''
    优化bagging的中基学习器的数量
    
    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        base_estimator : 数据类型为'object'
            基学习器对象
            该对象被认为是实现了sklearn estimator接口的对象。按需求直接使用sklearn中的类对象。
            例如需要使用knn回归来作为基学习器，就必须要导入KNeighborsRegressor这个类，并且将该类的对象赋值给estimator参数,base_estimator=KNeighborsRegressor()。
            具体学习器对象请看下面Notes

        n_estimators_start : 数据类型为'int'
            基学习器的数量。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            例如n_estimators_start = 1

            
        n_estimators_end : 数据类型为'int'
            基学习器的数量。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取最优参数。
            例如n_estimators_end = 200

        n_estimators_step : 数据类型为'int'
            每次测试时的步长。默认n_estimators_step = 5
            例如当步长n_estimators_step = 3时，左端点 n_estimators_start = 1，右端点n_estimators_end = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_jobs : int
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu
        
    Returns
    -------
        best_parm : 字典
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'n_estimators': 140}

        best_score : float
            模型交叉验证得分

    Notes
    --------
        knn作为基学习器，base_estimator = KNeighborsRegressor()
        决策树作为基学习器，base_estimator = DecisionTreeRegressor()
        随机森林作为基学习器，base_estimator = RandomForestRegressor()
        svm作为基学习器，base_estimator = SVR()

    '''
    parm_grid = {'n_estimators': np.arange(n_estimators_start, n_estimators_end, n_estimators_step)}
    best_parm, best_score = findBestParmByGridSearchCv(X, y, estimator=ensemble.BaggingRegressor(
        base_estimator=base_estimator), parm_grid=parm_grid, cv_num=cv_num, n_jobs=n_jobs)
    return best_parm, best_score


def adaBoostRegressionParm(X, y, n_estimators_start, n_estimators_end, n_estimators_step, max_depth_start,
                           max_depth_end, max_depth_step, cv_num=3, n_iter=20, n_jobs=-1):
    '''
    优化adaBoost中的基学习器CART树、基学习器的数量、学习率和loss函数
    
    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        n_estimators_start : 数据类型为'int'
            基学习器的数量。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            例如n_estimators_start = 1
            
        n_estimators_end : 数据类型为'int'
            基学习器的数量。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取最优参数。
            例如n_estimators_end = 200
        
        max_depth_start : 数据类型为'int'
            CART树的最大深度。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            例如max_depth_start = 1

        max_depth_end : 数据类型为'int'
            CART树的最大深度。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取最优参数。
            例如max_depth_end = 50
    
        max_depth_step : 数据类型为'int'
            每次测试时的步长。
            例如当步长n_estimators_step = 3时，左端点 n_estimators_start = 1，右端点n_estimators_end = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

        n_estimators_step : 数据类型为'int'
            每次测试时的步长。
            例如当步长n_estimators_step = 3时，左端点 n_estimators_start = 1，右端点n_estimators_end = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_iter : int
            训练次数，次数越大精度越高。默认n_iter=20

        n_jobs : int
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 字典
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'n_estimators': 152, 'learning_rate': 0.1, 'loss': 'linear'}

        best_score : float
            模型交叉验证得分

    '''
    dt_parm, dt_score = dtRegressionParm(X, y, max_depth_start, max_depth_end, max_depth_step)
    dt_model = DecisionTreeRegressor(max_depth=dt_parm['max_depth']).fit(X, y)
    parm_list = {'n_estimators': np.arange(n_estimators_start, n_estimators_end, n_estimators_step),
                 'learning_rate': [0.01, 0.05, 0.1, 0.5, 1, 1.5],
                 'loss': ['linear', 'square', 'exponential']}
    best_parm, best_score = findBestParmByRandomizedSearchCV(X, y, ensemble.AdaBoostRegressor(base_estimator=dt_model),
                                                             parm_list=parm_list, cv_num=cv_num, n_iter=n_iter,
                                                             n_jobs=n_jobs)
    best_parm['model'] = dt_model
    return best_parm, best_score


def gradientBoostingParm(X, y,
                         n_estimators_start, n_estimators_end,
                         max_depth_start, max_depth_end,
                         min_samples_leaf_start, min_samples_leaf_end,
                         min_samples_split_start, min_samples_split_end,
                         learning_rate, subsample,
                         max_depth_step, n_estimators_step, min_samples_leaf_step, min_samples_split_step,
                         cv_num=3, n_iter=20, n_jobs=-1):
    '''
    优化GBRT中基学习器的数量和CART树的深度

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        n_estimators_start : 数据类型为'int'
            基学习器(CART)的数量。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            例如n_estimators_start = 1

        n_estimators_end : 数据类型为'int'
            基学习器(CART)的数量。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取最优参数。
            例如n_estimators_end = 200
            
        max_depth_start : 数据类型为'int'
            CART树的最大深度。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            例如max_depth_start = 1

        max_depth_end : 数据类型为'int'
            CART树的最大深度。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取最优参数。
            例如max_depth_end = 50

        min_samples_leaf_start :  数据类型为'int'
            叶子节点最小样本数。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            建议min_samples_leaf一开始设置为1。如果样本量数量非常大，则推荐增大这个值。
            例如max_depth_start = 1

        min_samples_leaf_end : 数据类型为'int'
            叶子节点最小样本数。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            建议min_samples_leaf一开始设置为1。如果样本量数量非常大，则推荐增大这个值。
            例如max_depth_end = 2

        min_samples_split_start : 数据类型为'int'
            内部节点再划分所需最小样本数
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            建议min_samples_split一开始设置为2。如果样本量数量或者特征数非常大，则推荐增大这个值。
            例如max_depth_start = 2

        min_samples_split_end : 数据类型为'int'
            内部节点再划分所需最小样本数
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            建议min_samples_split一开始设置为2。如果样本量数量或者特征数非常大，则推荐增大这个值。
            例如max_depth_end = 3

        learning_rate : 数据类型为'类数组'(list or numpy.narray)
            每个弱学习器的学习步长。
            可以给定一个确定的量，例如learning_rate = [1.0]。
            如果不确定的话，也可以给定一个'类数组'，其中包含多个可能的取值，程序会自动选取最优值。例如learning_rate = [0.001, 0.01, 0.1, 1.0]。
            可以从10的幂次序列开始训练，根据效果微调。较小的ν意味着我们需要更多的弱学习器的迭代次数。

        subsample : 数据类型为'类数组'(list or numpy.narray)
            子采样。
            可以给定一个确定的量，例如subsample = [1.0]。
            如果不确定的话，也可以给定一个'类数组'，其中包含多个可能的取值，程序会自动选取最优值。例如subsample = [0.5, 0.6, 0.7, 0.8]。
            选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。推荐[0.5, 0.8]之间
        
        max_depth_step : 数据类型为'int'
            每次测试时的步长。
            例如当步长n_estimators_step = 3时，左端点 n_estimators_start = 1，右端点n_estimators_end = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

        n_estimators_step : 数据类型为'int'
            每次测试时的步长。
            例如当步长n_estimators_step = 3时，左端点 n_estimators_start = 1，右端点n_estimators_end = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

        min_samples_leaf_step : 数据类型为'int'
            每次测试时候的步长。默认min_samples_leaf_step = 1
            例如当步长min_samples_leaf_step = 3时，左端点min_samples_leaf_step = 1，右端点min_samples_leaf_step = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

        min_samples_split_step : 数据类型为'int'
            每次测试时候的步长。默认min_samples_split_step = 1
            例如当步长min_samples_split_step = 3时，左端点min_samples_split_step = 1，右端点min_samples_split_step = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_iter : int
            训练次数，次数越大精度越高。默认n_iter=20

        n_jobs : int
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 字典
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'subsample': 0.5, 'n_estimators': 116, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 29, 'loss': 'quantile', 'learning_rate': 0.1}

        best_score : float
            模型交叉验证得分

    '''
    parm_list = {'n_estimators': np.arange(n_estimators_start, n_estimators_end, n_estimators_step),
                 'max_depth': np.arange(max_depth_start, max_depth_end, max_depth_step),
                 'loss': ['ls', 'lad', 'huber', 'quantile'],
                 'learning_rate': learning_rate,
                 'subsample': subsample,
                 'min_samples_leaf': np.arange(min_samples_leaf_start, min_samples_leaf_end, min_samples_leaf_step),
                 'min_samples_split': np.arange(min_samples_split_start, min_samples_split_end, min_samples_split_step),
                 'max_features': ['log2', 'sqrt', None]}
    best_parm, best_score = findBestParmByRandomizedSearchCV(X, y, ensemble.GradientBoostingRegressor(),
                                                             parm_list=parm_list, cv_num=cv_num, n_iter=n_iter,
                                                             n_jobs=n_jobs)
    return best_parm, best_score


def lassoParm(X, y, alpha, max_iter, cv_num=3, n_iter=20, n_jobs=-1):
    '''
    lasso回归参数调优

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        alpha : float 或者 '类数组'
            正则项系数。数值越大，则对复杂模型的惩罚力度越大。'类数组'是拥有数组结构，可以转化为narray的任何python对象。
            传入数组序列时，程序会在其中自动选优
            
        max_iter : int
            最大迭代次数

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_iter : int
            训练次数，次数越大精度越高。默认n_iter=20

        n_jobs : int
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 字典
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'alpha': 0.1}

        best_score : float
            模型交叉验证得分

    '''
    parm_list = {'alpha': alpha}
    best_parm, best_score = findBestParmByRandomizedSearchCV(X, y, Lasso(max_iter=max_iter), parm_list=parm_list,
                                                             cv_num=cv_num, n_iter=n_iter, n_jobs=n_jobs)
    return best_parm, best_score


def pearsonrValidation(X, y, X_name=None):
    '''
    皮尔逊相关系数

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            自变量
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            因变量
            例如y_train = [16.50000,31.10000,10.50000]
        
        X_name : 形状为(特征数量, )的'类数组'。可选
            特征名。默认X_name = None

    Returns
    -------
        retVal : DataFrame
            若给没有给定参数X_name时候，返回一个(特征数量, 2)的DataFrame，第一列为皮尔逊系数，第二列为p值。
            若给定参数X_name时候，返回一个(特征数量, 3)的DataFrame，首列为特征名，第二列为皮尔逊系数，第三列为p值。

    '''
    res = []
    for i in range(X.shape[1]):
        res.insert(i, pearsonr(X[:, i], y))
    res = np.array(res)
    transVal = {'pearsonrCoef': res[:, 0],
                'pVal': res[:, 1]}
    retVal = DataFrame(transVal)
    if X_name is not None:
        # retVal.loc[:, 'd'] = X_name
        retVal.insert(loc=0, column='featureName', value=X_name)
    return retVal


def modelsRoc(modelDict, X_test, y_test):
    '''
    绘制模型的ROC曲线，计算模型的AUC值
    
    Parameters
    ----------
        modelDict : 字典
            key为模型名字，value为训练好的模型
            例如modelDict = {'svr': svr}
            
        X_test : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            自变量
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]
            
        y_test : 数据类型为'类数组'(list or numpy.narray)
            类别标签
            例如y_train = [1,0,1]

    '''
    fig, ax = plt.subplots(figsize=(12, 10))
    for m_name, model in modelDict.item():
        roc = plot_roc_curve(model, X_test, y_test, ax, linewidth=1)
    plt.show()


def featureCorrelation(X, y, X_name=None, img=None):
    '''
    相关性分析

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            自变量
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            因变量
            例如y_train = [16.50000,31.10000,10.50000]

        X_name : 形状为(特征数量, )的'类数组'。可选
            特征名。默认X_name = None

        img : boolean
            如果需要将自变量和因变量的线性关系可视化，请让img=True，否则请保持默认img=None。

    '''
    ps = pearsonrValidation(X, y, X_name)
    print(ps)
    if img is not None:
        for i in range(X.shape[1]):
            plt.scatter(X[:, i], y)
            if X_name is None:
                plt.show()
            else:
                plt.title(X_name[i])
                plt.show()


def arimaModel(data, d, pmax, qmax):
    '''
    ARIMA模型。模型定阶选用BIC检验，组合各种p和q，获取最小BIC值的p和q

    Parameters
    ----------
        data : 数据类型为'numpy.narray'
            原始数据序列。为一维数组。为非平稳序列
            每一个参数为该时间段的值。
            例如# data=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422]

        d : 数据类型为'int'
            差分阶数。
            将序列变成平稳性序列。推荐使用该函数前先对原始数据进行差分，观测数据平稳性。

        pmax : 数据类型为'int'
            AR(p)的最大值。一般不超过（data长度 / 10）

        qmax : 数据类型为'int'
            MA(q)的最大值。一般不超过（data长度 / 10）

    Returns
    -------
        model : object
            训练好的ARIMA模型。可以通过model.forecast(step)来进行预测，返回的结果包括：预测结果、标准误差、置信区间

    Examples
    -------
        data=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422,
                6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355,
                10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767,
                12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232,
                13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248,
                9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722,
                11999,9390,13481,14795,15845,15271,14686,11054,10395]
        data = np.array(data, dtype=np.float)
        arimamodel = arimaModel(data, 1, np.int(data.__len__() / 10), np.int(data.__len__() / 10))
        # 想要预测未来五个时间单位的值
        arimamodel.forecast(5)

    '''
    from statsmodels.tsa.arima_model import ARIMA
    data = np.array(pd.DataFrame(data).dropna()).flatten()
    bic_matrix = []
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(ARIMA(data, (p, d, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    bic_matrix = pd.DataFrame(bic_matrix)
    p, q = bic_matrix.stack().astype(float).idxmin()
    model = ARIMA(data, (p, d, q)).fit()
    # print(model.forecast(5))
    return model


def armaModel(data, pmax, qmax):
    '''
    ARMA模型。模型定阶选用BIC检验，组合各种p和q，获取最小BIC值的p和q

    Parameters
    ----------
        data : 数据类型为'numpy.narray'
            原始数据序列。为一维数组。要求为平稳序列
            每一个参数为该时间段的值。
            例如# data=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422]

        pmax : 数据类型为'int'
            AR(p)的最大值。一般不超过（data长度 / 10）

        qmax : 数据类型为'int'
            MA(q)的最大值。一般不超过（data长度 / 10）

    Returns
    -------
        model : object
            训练好的ARMA模型。可以通过model.forecast(step)来进行预测，返回的结果包括：预测结果、标准误差、置信区间

    Examples
    -------
        data=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422,
                6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355,
                10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767,
                12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232,
                13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248,
                9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722,
                11999,9390,13481,14795,15845,15271,14686,11054,10395]
        data = np.array(data, dtype=np.float)
        armamodel = armaModel(np.diff(data, 1), np.int(data.__len__() / 10), np.int(data.__len__() / 10))
        # 想要预测未来五个时间单位的值
        arimamodel.forecast(5)

    '''
    from statsmodels.tsa.arima_model import ARMA
    data = np.array(pd.DataFrame(data).dropna()).flatten()
    bic_matrix = []
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(ARMA(data, (p, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    bic_matrix = pd.DataFrame(bic_matrix)
    p, q = bic_matrix.stack().astype(float).idxmin()
    model = ARMA(data, (p, q)).fit()
    # print(model.forecast(5))
    return model


class GraySystem:
    '''
    该类主要包括灰色关联度分析和多因素灰色预测模型

    '''

    def AGO(self, m):
        '''
        累加生成算子

        Parameters
        ----------
            m : 数据类型为'numpy.narray'
                为一维数组
                需要进行累加操作的序列。
                例如m = [104, 101.8, 105.8, 111.5, 115.97, 120.03, 113.3, 116.4, 105.1, 83.4, 73.3]

        Returns
        -------
            li : 数据类型为'numpy.narray'
                为一维数组
                累加完成后的序列
                x0 = [104, 101.8, 105.8, 111.5, 115.97, 120.03, 113.3, 116.4, 105.1, 83.4, 73.3]

        '''
        li = np.zeros(shape=m.shape)
        for i in range(m.__len__()):
            temp = 0
            for j in range(i + 1):
                temp = m[j] + temp
            li[i] = temp
        return li

    def MEAN(self, m):
        '''
        均值生成算子

        Parameters
        ----------
            m : 数据类型为'numpy.narray'
                为一维数组
                需要生成邻均值序列的序列。
                例如m = [104, 101.8, 105.8, 111.5, 115.97, 120.03, 113.3, 116.4, 105.1, 83.4, 73.3]

        Returns
        -------
            li : 数据类型为'numpy.narray'
                为一维数组
                邻均值序列
                x0 = [104, 101.8, 105.8, 111.5, 115.97, 120.03, 113.3, 116.4, 105.1, 83.4, 73.3]

        '''
        li = np.zeros(shape=m.shape)
        for i in range(m.__len__()):
            if i == 0:
                continue
            li[i] = 0.5 * (m[i] + m[i - 1])
        li = np.delete(li, 0)
        return li

    def greyRelationAnalysis(self, X, y):
        '''
        灰色关联度分析

        Parameters
        ----------
        X : 数据类型为'numpy.narray'
            比较队列。列向量表示X = [x0, x1, x2,]
            数组每一列都是一条相关因素序列，每一行代表某个时间节点该因素的值。
            列向量表示X = [x0, x1, x2, x3]
            即X = [[104.00000,135.60000,131.60000,54.20000],
                    [101.80000,140.20000,135.50000,54.90000],
                    [105.80000,140.10000,142.60000,54.80000],
                    [111.50000,146.90000,143.20000,56.30000],
                    [115.97000,144.00000,142.20000,54.50000]]

        y : 数据类型为'numpy.narray'
            参考队列。列向量
            每一个值都是目标在该时间节点，多个因素影响下的结果
            例如y = [[560823],
                    [542386],
                    [604834],
                    [591248],
                    [583031]]

        Returns
        -------
            r : 形状为(特征数量, )的'numpy类型数组'
                关联度数组
                例如r = [0.72135605 0.69545271 0.65791122 0.73256354]
        
        Examples
        -------
            a = [560823, 542386, 604834, 591248, 583031, 640636, 575688, 689637, 570790, 519574, 614677]
            x0 = [104, 101.8, 105.8, 111.5, 115.97, 120.03, 113.3, 116.4, 105.1, 83.4, 73.3]
            x1 = [135.6, 140.2, 140.1, 146.9, 144, 143, 133.3, 135.7, 125.8, 98.5, 99.8]
            x2 = [131.6, 135.5, 142.6, 143.2, 142.2, 138.4, 138.4, 135, 122.5, 87.2, 96.5]
            x3 = [54.2,54.9,54.8,56.3,54.5,54.6,54.9,54.8,49.3,41.5,48.9]
            r = gs.greyRelationAnalysis(np.array(mat([x0, x1, x2, x3]).T), np.array(mat(a).T))
            # print(r)结果为r = [0.72135605 0.69545271 0.65791122 0.73256354]

        '''
        y = np.array(y, dtype=float).ravel()
        X_mean = X.mean(axis=0)
        y_mean = y.mean(axis=0)
        for i in range(X.shape[1]):
            X[:, i] = X[:, i] / X_mean[i]
        for i in range(y.shape[0]):
            y[i] = y[i] / y_mean
        new = []
        for i in range(X.shape[1]):
            new.append(np.array(mat(X[:, i] - y)))
        new = np.array(mat(np.array(new)).T)
        # new = np.array(mat(new).T)
        mmax = np.max(np.max(new.__abs__(), axis=0))
        mmin = np.min(np.min(new.__abs__(), axis=0))
        rho = 0.5
        # 关联系数
        ksi = ((mmin + rho * mmax) / (new.__abs__() + rho * mmax))
        # 关联度
        r = sum(ksi, axis=0) / ksi.shape[0]
        return r

    def greyPredictiveParm(self, X_original, y_original):
        '''
        获得发展系数和驱动系数

        Parameters
        ----------
            X_original : 形状为(相关因素序列数量， 每个因素包含的数量)的numpy类型数组
                相关因素。
                数组的每一行代表一条相关因素序列，每一列代表某个时间节点该因素的值。

            y_original : 形状为(系统特征数据数量, )的numpy类型数组
                系统特征数据序列(真实值)。
                每一个值都是目标在该时间节点，多个因素影响下的结果

        Returns
        -------
            al :  float
                GM(1, n)模型中的发展系数

            b : 形状为(相关因素个数， )的numpy类型数组
                GM(1, n)模型中的驱动系数列表。
                每一个值代表对应相关因素序列的驱动系数。

        '''
        xi = []
        for i in range(X_original.shape[0]):
            xi.append(self.AGO(X_original[i, :]))
        xi = np.array(xi)
        z = self.MEAN(np.array(self.AGO(y_original)))
        Y = y_original.reshape(-1, 1)
        Y = np.delete(Y, 0)
        Y = mat(Y)
        B = np.array(mat(xi[:, 1:]).T)
        B = np.insert(B, 0, -z, 1)
        B = mat(B)
        theat = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y.T)
        al = theat[0, 0]
        b = theat[1:, :].T
        b = np.array(b).flatten()
        return al, b

    def greyPredictiveModel(self, X_new, al, b, y_first):
        '''
        多因素灰色预测模型

        Parameters
        ----------
            X_new : 数据类型为'numpy.narray'
                新的相关因素。列向量表示X = [x0, x1, x2,]
                数组每一列都是一条相关因素序列，每一行代表某个时间节点该因素的值。
                列向量表示X = [x0, x1, x2, x3]
                即X = [[104.00000,135.60000,131.60000,54.20000],
                        [101.80000,140.20000,135.50000,54.90000],
                        [105.80000,140.10000,142.60000,54.80000],
                        [111.50000,146.90000,143.20000,56.30000],
                        [115.97000,144.00000,142.20000,54.50000]]

            al : float
                GM(1, n)模型中的发展系数

            b : 形状为(相关因素个数， )的numpy类型数组
                GM(1, n)模型中的驱动系数列表。
                每一个值代表对应相关因素序列的驱动系数。

            y_first : float
                预测值的起点

        Returns
        -------
            G : 形状为(系统特征数据数量, )的numpy类型数组
                系统特征数据序列(预测值)。
                每一个值都是目标在该时间节点，多个因素影响下的结果。
        
        Examples
        -------
            # 根据已知数据得出参数al, b
            a = [560823, 542386, 604834, 591248, 583031, 640636, 575688, 689637, 570790, 519574, 614677]
            x0 = [104, 101.8, 105.8, 111.5, 115.97, 120.03, 113.3, 116.4, 105.1, 83.4, 73.3]
            x1 = [135.6, 140.2, 140.1, 146.9, 144, 143, 133.3, 135.7, 125.8, 98.5, 99.8]
            x2 = [131.6, 135.5, 142.6, 143.2, 142.2, 138.4, 138.4, 135, 122.5, 87.2, 96.5]
            x3 = [54.2,54.9,54.8,56.3,54.5,54.6,54.9,54.8,49.3,41.5,48.9]
            al, b = gs.greyPredictiveParm(np.array([x0, x1, x2, x3]), np.array(a))

            # 当有新的相关因素产生时可以预测
            x_0 = [113.57, 112.03, 128.3, 106.4, 113.1, 71.4, 92.5]
            x_1 = [139.2, 135.2, 145.1, 141.5, 142, 141.6, 136.3]
            x_2 = [138, 131.5, 142.6, 138.4, 142.2, 131, 134.3]
            x_3 = [56.1,55.9,51.8,58.3,53.4,56.7,56.2]
            G = gs.greyPredictiveModel(np.array([x_0, x_1, x_2, x_3]), al, b, a[len(a) - 1])

        '''
        xi = []
        for i in range(X_new.shape[0]):
            xi.append(self.AGO(X_new[i, :]))
        xi = np.array(xi)
        U = []
        for j in range(xi.shape[1]):
            sum1 = 0
            for i in range(xi.shape[0]):
                sum1 += mat(xi)[i, j] * b[i]
            U.append(sum1)
        F = []
        f = 1
        F.append(y_first)
        while f < X_new.shape[1]:
            F.append((y_first - U[f] / al) * np.exp(-al * f) + U[f] / al)
            f = f + 1
        F = np.array(F).flatten()
        G = [y_first]
        g = 1
        while g < X_new.shape[1]:
            G.append(F[g] - F[g - 1])
            g += 1
        G = np.array(G)
        return G


def find_max_region(file, mbKSize):
    '''
    最大连通区域
    :param file:
    :param mbKSize:
    :return:
    '''
    import cv2
    img = cv2.cvtColor(file, code=cv2.COLOR_RGB2GRAY)
    img7 = cv2.medianBlur(src=img, ksize=mbKSize)
    ret, threshold = cv2.threshold(img7, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold, connectivity=8)
    labelNum = 0
    for i in range(stats.shape[0]):
        if stats[i, 0] == 0 and stats[i, 1] == 0:
            labelNum = i
    stats = np.delete(stats, [0, 1], axis=0)
    num_labels = num_labels - 2
    # 将label列表中labelNum组全部置为0背景
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i][j] == labelNum:
                labels[i][j] = 0
    output = np.zeros((threshold.shape[0], threshold.shape[1]), np.uint8)
    count_list = []
    for i in range(1, num_labels + 1):
        mask = labels == i
        output[:, :][mask] = 255
        # 计算二值图中值为255的像素点数量
        coutNum = 0
        for w in range(output.shape[0]):
            for h in range(output.shape[1]):
                if output[w][h] == 255:
                    coutNum += 15
        count_list.append(coutNum)
    # 将面积最大的下标记录,第maxIdx个轮廓就是最大连通图
    maxIdx = np.argmax(count_list) + 1
    mask = labels == maxIdx
    output[:, :][mask] = 255
    return output


def binaryMask(mask_file):
    '''
    掩膜二值化
    :param mask_file:
    :return:
    '''
    for i in range(mask_file.shape[0]):
        for j in range(mask_file.shape[1]):
            mask_file[i][j] = 1 if (mask_file[i][j] == 255) else 0


def bow(descriptor_list, k, label):
    '''
    词袋算法
    :param descriptor_list: 图片数据集的特征描述子列表
    :param k: 要将特征描述子分成几类
    :param label: 图片标签
    :return:
    '''
    ret = []
    for item in descriptor_list:
        item = item.reshape(-1, 1)
        # 先测出一个item的最佳k值
        y_pre = KMeans(n_clusters=k, random_state=32).fit_predict(item)
        l = len(y_pre)
        unique, counts = np.unique(y_pre, return_counts=True)
        frequency = dict(zip(unique, counts / l))
        ret.append(list(frequency.values()))
    ret = np.insert(np.array(mat(ret)), k, label, axis=1)
    return ret


def surfaceFitting(img, k, deg=3, plot=False):
    '''
    表面拟合
    Parameters
    ----------
        img：数据类型为numpy.narray
            图片数据

        k：数据类型为int
            提取拟合数据点用到的阈值k
            将图片中低于k的灰度值置为0。

        deg:数据类型为int
            拟合使用的多项式的阶数
            默认为3

        plot：数据类型为boolean
            是否展示绘制拟合的结果。
            默认为False

    Returns
    -------
        surfacePosition:数据类型为numpy.array
            表面位置数组
            其中的每一个数 value,有这样一个关系：col_index = img.shape[1] - value

        ret:数据类型为numpy.narray
            拟合点的图片

    '''
    ret = np.copy(img)
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            if ret[i][j] <= (k):
                ret[i][j] = 0
    temp = []
    for i in range(ret.shape[1]):
        fd = np.flipud(ret[:, i])
        j = ret.shape[1]
        while j > 0:
            if j != 0 and j < 500 and fd[j] != 0:
                temp.append((512 - j, i + 1))
                break
            j -= 1
    temp = np.array(temp)
    X = temp[:, 1]
    y = temp[:, 0]
    z1 = np.polyfit(X, y, deg)
    p1 = np.poly1d(z1)
    X_test = [i + 1 for i in range(ret.shape[1])]
    surfacePosition = np.array(p1(X_test), dtype=np.int32)
    if plot:
        pts1 = np.concatenate((X_test, surfacePosition)).reshape((2, 512)).T
        test = cv2.polylines(np.copy(img), [pts1], False, color=(255, 0, 0))
        cv2.imshow('press to continue', test)
        cv2.waitKey(0)
    return surfacePosition, ret


def flatten(img, surfacePosition):
    '''
    展平图片
    Parameters
    ----------
        img:数据类型为numpy.narray
            图片数据

        surfacePosition：数据类型为numpy.array
            表面位置索引数组
            由列向量索引组成，长度等于图片的列向量数量
            例如np.array([224, 212, 264, 203 ....，123])

    Returns
    -------
        ret：数据类型为numpy.narray
            展平后的图片
    '''
    ret = np.zeros(shape=img.shape, dtype=np.uint8)
    imax = np.max(surfacePosition)
    imin = np.min(surfacePosition)
    mid = int((imax + imin) / 2)
    for idx, value in enumerate(surfacePosition):
        diff = np.abs(value - mid)
        temp = img[:, idx]
        if value > mid:
            k = diff
            for j in range(temp.size - diff):
                ret[k][idx] = temp[j]
                k += 1
        else:
            k = 0
            for j in range(diff, temp.size):
                ret[k][idx] = temp[j]
                k += 1
    return ret


def denoise(img, n, kernel_size=3):
    '''
    图片去噪
    Parameters
    ----------
        img:数据类型为numpy.narray
            图片数据

        n:数据类型为int
            使用前n行的均值和标准差进行阈值去噪

        kernel_size：数据类型为int
            中值滤波核的大小

    Returns
    -------
        ret:数据类型为numpy.narray
            返回图片

    '''
    ret = cv2.medianBlur(img, kernel_size)
    k = []
    num = 0
    sum = 0
    for i in range(n):
        for j in range(ret.shape[1]):
            if ret[i][j] <= (5):
                continue
            sum += ret[i][j]
            k.append(ret[i][j])
            num += 1
    ave = num / sum
    sum = 0
    for value in k:
        sum += np.power(value - ave, 2)
    sd = np.sqrt(sum / num)
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            if ret[i][j] <= sd + ave:
                ret[i][j] = 0
    return ret


def cropImg(img, top=0, button=0, left=0, right=0):
    '''
    裁剪图片
    Parameters
    ----------
        img:数据类型为numpy.narray
            图片数据

        top:数据类型为int
            裁剪掉距离顶部的k行
            默认值为0

        button:数据类型为int
            裁剪掉距离底部的k行
            默认值为0

        left:数据类型为int
            裁剪掉距离左边的k列
            默认值为0

        right:数据类型为int
            裁剪掉距离右边的k列
            默认值为0

    Returns
    -------
        ret:数据类型为numpy.narray
            裁剪完的图片

    Examples
    --------
        img = cv2.imread('7-462.jpg', flags=0)
        ret = cropImg(img, 60)
        cv2.imshow('test', ret)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''
    ret = img[top:img.shape[0] - button, left:img.shape[1] - right]
    return ret



