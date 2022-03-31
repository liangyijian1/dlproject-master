from src.utils.utils import *


class Regression:
    """
    该类包含各种回归方法。
    多元线性回归、svm回归、决策树回归、随机森林回归、knn回归、bagging回归、adaBoost回归、GBRT回归、lasso回归、非线性模型回归

    """

    def mlRegression(self, X_train, y_train):
        """
        多元线性回归。

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

        Returns
        -------
            coef : 数据类型为'类数组'(list or numpy.narray)
                回归模型系数

            intercept : 数据类型为'float'
                回归模型截距

        """
        from sklearn import linear_model
        linreg = linear_model.LinearRegression()
        linreg.fit(X_train, y_train)
        intercept = linreg.intercept_
        coef = linreg.coef_
        return linreg, coef, intercept

    def svmRegression(self, X_train, y_train, C, gama, epsilon, cv_num=3, n_iter=20):
        """
        svm回归。
        本函数的超参数可以给定一个确定的值，也可以让程序自动寻优，具体见下方对三个超参数的注释。

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

        Returns
        -------
            model : 数据类型为'object'
                训练好的模型。
                可以调用model.predict()函数可以对新的数据进行预测，具体使用方法可以见下方Examples
                可以使用utils.py中的saveModel函数将模型以.pkl的文件保存到本地。

            best_parm : 数据类型为'字典'
                模型最优参数。key为参数名称，value为对应值。
                例如best_parm = {'C': 10.0, 'epsilon': 0.0, 'gamma': 0.7142857142857143}

            best_score : 数据类型为'float'
                模型交叉验证得分

        Examples
        --------
            svr, svr_parm, cv_score = regr.svmRegression(X_train, y_train, C=[0.01, 0.1, 1.0, 10], gama=[0.01, 0.1, 1.0, 10], epsilon=[0.01, 0.1, 1.0, 10], cv_num=5, n_iter=20)
            # 直接使用predict()函数进行预测
            y_pre = svr.predict(X_test)
            # 使用utils.py中的saveModel()函数将模型保存到本地
            saveModel('svr.pkl', svr)
            # 使用utils.py中的loadModel()函数来调用本地模型，并进行预测
            y_pre = loadModel('svr.pkl').predict(X_test)

        Notes
        --------
            自动调参默认开启全部CPU，如果需要更改，请修改源程序中svmParm()函数中的n_jobs参数，当n_jobs = -1时启用全部cpu

        """
        from sklearn import svm
        best_parm, best_score = svrParm(X_train, y_train, C=C, gama=gama, epsilon=epsilon, cv_num=cv_num, n_iter=n_iter)
        model = svm.SVR(C=best_parm['C'], kernel='rbf', gamma=best_parm['gamma'], epsilon=best_parm['epsilon']).fit(
            X_train, y_train)
        return model, best_parm, best_score

    def dtRegression(self, X_train, y_train, max_depth_start, max_depth_end, max_depth_step=1, cv_num=3, n_iter=20):
        """
        决策树回归。
        本函数超参数只需要设定树的最大深度，当使用默认步长1时，max_depth_start, max_depth_end分别等于1和50时候
        树的最大深度max_depth就在[1, 50)中取最优值。

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

        Returns
        -------
            model : 数据类型为'object'
                训练好的模型。
                可以调用model.predict()函数可以对新的数据进行预测，具体使用方法可以见下方Examples
                可以使用utils.py中的saveModel函数将模型以.pkl的文件保存到本地。

            best_parm : 数据类型为'字典'
                模型最优参数。key为参数名称，value为对应值。
                例如best_parm = {'max_depth': 16}

            best_score : 数据类型为'float'
                模型交叉验证得分

        Examples
        --------
            dt, dt_parm, dt_score = regr.dtRegression(X_train, y_train, max_depth_start=1, max_depth_end=50, max_depth_step=1, cv_num=4, n_iter=20)
            # 直接使用predict()函数进行预测
            y_pre = dt.predict(X_test)
            # 使用utils.py中的saveModel()函数将模型保存到本地
            saveModel('dt.pkl', dt)
            # 使用utils.py中的loadModel()函数来调用本地模型，并进行预测
            y_pre = loadModel('dt.pkl').predict(X_test)

        Notes
        --------
            自动调参默认开启全部CPU，如果需要更改，请修改源程序中dtRegressionParm函数中的n_jobs参数，当n_jobs = -1时启用全部cpu

        """
        from sklearn.tree import DecisionTreeRegressor
        best_parm, best_score = dtRegressionParm(X_train, y_train, max_depth_start, max_depth_end, max_depth_step,
                                                 cv_num=cv_num, n_iter=n_iter)
        model = DecisionTreeRegressor(max_depth=best_parm['max_depth']).fit(X_train, y_train)
        return model, best_parm, best_score

    def rfRegression(self, X_train, y_train,
                     n_tree_start, n_tree_end,
                     max_depth_start, max_depth_end,
                     min_samples_leaf_start, min_samples_leaf_end,
                     max_depth_step=1, min_samples_leaf_step=2, n_tree_step=10, cv_num=3, n_iter=20):
        """
        随机森林回归
        本函数超参数只需要设定随机森林中树的棵树，树的最大深度和叶子节点最少的样本数。

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
                例如当步长min_samples_leaf_step = 3时，左端点min_samples_leaf_step = 1，右端点min_samples_leaf_step = 10，此时 叶子节点最少的样本数就在[1, 4, 7]中取一个较优值。

            cv_num : 数据类型为'int'
                S折交叉验证的折数。默认cv_num = 3
                即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

            n_iter : 数据类型为'int'
                训练次数，次数越大精度越高。默认n_iter=20

        Returns
        -------
            model : object
                训练好的模型。
                可以调用model.predict()函数可以对新的数据进行预测，具体使用方法可以见下方Examples
                可以使用utils.py中的saveModel函数将模型以.pkl的文件保存到本地。

            best_parm : 字典
                模型最优参数。key为参数名称，value为对应值。
                例如best_parm = {'n_tree': 191, 'max_depth': 14, 'min_samples_leaf': '1'}

            best_score : float
                模型交叉验证得分

        Examples
        --------
            rf, rf_parm, rf_score = regr.rfRegression(X_train, y_train, 1, 200, 1, 50, 1, 100)
            # 直接使用predict()函数进行预测
            y_pre = rf.predict(X_test)
            # 使用utils.py中的saveModel()函数将模型保存到本地
            saveModel('rf.pkl', rf)
            # 使用utils.py中的loadModel()函数来调用本地模型，并进行预测
            y_pre = loadModel('rf.pkl').predict(X_test)

        Notes
        --------
            自动调参默认开启全部CPU，如果需要更改，请修改源程序中rfRegressionParm函数中的n_jobs参数，当n_jobs = -1时启用全部cpu

        """
        from sklearn import ensemble
        best_parm, best_score = rfRegressionParm(X_train, y_train,
                                                 n_tree_start, n_tree_end, n_tree_step,
                                                 max_depth_start, max_depth_end, max_depth_step,
                                                 min_samples_leaf_start, min_samples_leaf_end, min_samples_leaf_step,
                                                 cv_num, n_iter)
        model = ensemble.RandomForestRegressor(n_estimators=best_parm['n_estimators'],
                                               max_depth=best_parm['max_depth'],
                                               max_features=best_parm['max_features'],
                                               min_samples_leaf=best_parm['min_samples_leaf']).fit(X_train, y_train)
        return model, best_parm, best_score

    def knnRegression(self, X_train, y_train, n_neighbors_start, n_neighbors_end, n_neighbors_step=1, cv_num=3):
        '''
        knn回归
        本函数超参数只需设定k邻近的值
        
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
 

        Returns
        -------
            model : 数据类型为'object'
                训练好的模型。
                可以调用model.predict()函数可以对新的数据进行预测，具体使用方法可以见下方Examples
                可以使用utils.py中的saveModel函数将模型以.pkl的文件保存到本地。

            best_parm : 数据类型为'字典'
                模型最优参数。key为参数名称，value为对应值。
                例如best_parm = {'n_neighbors': 9}

            best_score : 数据类型为'float'
                模型交叉验证得分

        Examples
        --------
            knn, knn_parm, knn_score = regr.knnRegression(X_train, y_train, 1, 50)
            # 直接使用predict()函数进行预测
            y_pre = knn.predict(X_test)
            # 使用utils.py中的saveModel()函数将模型保存到本地
            saveModel('svm.pkl', knn)
            # 使用utils.py中的loadModel()函数来调用本地模型，并进行预测
            y_pre = loadModel('svm.pkl').predict(X_test)

        Notes
        --------
            自动调参默认开启全部CPU，如果需要更改，请修改源程序中knnRegressionParm函数中的n_jobs参数，当n_jobs = -1时启用全部cpu

        '''
        from sklearn.neighbors import KNeighborsRegressor
        best_parm, best_score = knnRegressionParm(X_train, y_train, n_neighbors_start, n_neighbors_end,
                                                  n_neighbors_step, cv_num=cv_num, n_jobs=-1)
        model = KNeighborsRegressor(n_neighbors=best_parm['n_neighbors']).fit(X_train, y_train)
        return model, best_parm, best_score

    def baggingRegression(self, X_train, y_train, base_estimator, n_estimators_start, n_estimators_end,
                          n_estimators_step=5, cv_num=3):
        '''
        bagging回归
        本函数需要设定的超参数时基学习器和它的数量
        
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

        Returns
        -------
            model : 数据类型为'object'
                训练好的模型。
                可以调用model.predict()函数可以对新的数据进行预测，具体使用方法可以见下方Examples
                可以使用utils.py中的saveModel函数将模型以.pkl的文件保存到本地。

            best_parm : 数据类型为'字典'
                模型最优参数。key为参数名称，value为对应值。
                例如best_parm = {'n_estimators': 86, 'loss': 'exponential', 'learning_rate': 0.1, 'model': DecisionTreeRegressor(max_depth=3)}

            best_score : 数据类型为'float'
                模型交叉验证得分

        Examples
        --------
            bag, bag_parm, bag_score = regr.baggingRegression(X_train, y_train, base_estimator=KNeighborsRegressor(), n_estimators_start=1, n_estimators_end=200)
            # 直接使用predict()函数进行预测
            y_pre = bag.predict(X_test)
            # 使用utils.py中的saveModel()函数将模型保存到本地
            saveModel('bag.pkl', bag)
            # 使用utils.py中的loadModel()函数来调用本地模型，并进行预测
            y_pre = loadModel('bag.pkl').predict(X_test)                

        Notes
        --------
            自动调参默认开启全部CPU，如果需要更改，请修改源程序中baggingRegressionParm函数中的n_jobs参数，当n_jobs = -1时启用全部cpu
            knn作为基学习器，base_estimator = KNeighborsRegressor()
            决策树作为基学习器，base_estimator = DecisionTreeRegressor()
            随机森林作为基学习器，base_estimator = RandomForestRegressor()
            svm作为基学习器，base_estimator = SVR()

        '''
        from sklearn import ensemble
        best_parm, best_score = baggingRegressionParm(X_train, y_train, base_estimator, n_estimators_start,
                                                      n_estimators_end, n_estimators_step, cv_num, -1)
        model = ensemble.BaggingRegressor(base_estimator=base_estimator,
                                          n_estimators=best_parm['n_estimators']).fit(X_train, y_train)
        return model, best_parm, best_score

    def adaBoostRegression(self, X_train, y_train, n_estimators_start, n_estimators_end, max_depth_start, max_depth_end,
                           max_depth_step=1, n_estimators_step=5, cv_num=3, n_iter=20):
        '''
        adaBoost回归。使用CART作为基学习器
        本函数需要设定的超参数是CART树的深度，和基学习器的数量
        
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
                每次测试时的步长。默认max_depth_step = 1
                例如当步长n_estimators_step = 3时，左端点 n_estimators_start = 1，右端点n_estimators_end = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

            n_estimators_step : 数据类型为'int'
                每次测试时的步长。默认n_estimators_step = 5
                例如当步长n_estimators_step = 3时，左端点 n_estimators_start = 1，右端点n_estimators_end = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

            cv_num : 数据类型为'int'
                S折交叉验证的折数。默认cv_num = 3
                即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

            n_iter : 数据类型为'int'
                训练次数，次数越大精度越高。默认n_iter=20

        Returns
        -------
            model : 数据类型为'object'
                训练好的模型。
                可以调用model.predict()函数可以对新的数据进行预测，具体使用方法可以见下方Examples
                可以使用utils.py中的saveModel函数将模型以.pkl的文件保存到本地。

            best_parm : 数据类型为'字典'
                模型最优参数。key为参数名称，value为对应值。
                例如best_parm = {'n_estimators': 86, 'loss': 'exponential', 'learning_rate': 0.1, 'model': DecisionTreeRegressor(max_depth=3)}

            best_score : 数据类型为'float'
                模型交叉验证得分

        Examples
        --------
            ada, ada_parm, ada_score = regr.knnRegression(X_train, y_train, 1, 200, 1, 50)
            # 直接使用predict()函数进行预测
            y_pre = ada.predict(X_test)
            # 使用utils.py中的saveModel()函数将模型保存到本地
            saveModel('ada.pkl', ada)
            # 使用utils.py中的loadModel()函数来调用本地模型，并进行预测
            y_pre = loadModel('ada.pkl').predict(X_test)                

        Notes
        --------
            自动调参默认开启全部CPU，如果需要更改，请修改源程序中adaBoostRegressionParm函数中的n_jobs参数，当n_jobs = -1时启用全部cpu

        '''
        from sklearn import ensemble
        best_parm, best_score = adaBoostRegressionParm(X_train, y_train, n_estimators_start, n_estimators_end,
                                                       n_estimators_step, max_depth_start, max_depth_end,
                                                       max_depth_step, cv_num, n_iter, -1)
        model = ensemble.AdaBoostRegressor(base_estimator=best_parm['model'], n_estimators=best_parm['n_estimators'],
                                           learning_rate=best_parm['learning_rate'], loss=best_parm['loss']).fit(
            X_train, y_train)
        return model, best_parm, best_score

    def gradientBoostRegression(self, X_train, y_train,
                                n_estimators_start, n_estimators_end,
                                max_depth_start, max_depth_end,
                                min_samples_leaf_start, min_samples_leaf_end,
                                min_samples_split_start, min_samples_split_end,
                                learning_rate, subsample,
                                max_depth_step=1, n_estimators_step=5, min_samples_leaf_step=1,
                                min_samples_split_step=1,
                                cv_num=3, n_iter=20):
        '''
        GBRT回归
        本函数需要调整的参数是基学习器(CART)的数量、CART树的最大深度、叶子节点最小样本数、内部节点再划分所需最小样本数、每个弱学习器的学习步长、子采样
        
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
                每次测试时的步长。默认max_depth_step = 1
                例如当步长n_estimators_step = 3时，左端点 n_estimators_start = 1，右端点n_estimators_end = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

            n_estimators_step : 数据类型为'int'
                每次测试时的步长。默认n_estimators_step = 5
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

            n_iter : 数据类型为'int'
                训练次数，次数越大精度越高。默认n_iter=20

        Returns
        -------
            model : 数据类型为'object'
                训练好的模型。
                可以调用model.predict()函数可以对新的数据进行预测，具体使用方法可以见下方Examples
                可以使用utils.py中的saveModel函数将模型以.pkl的文件保存到本地。

            best_parm : 数据类型为'字典'
                模型最优参数。key为参数名称，value为对应值。
                例如best_parm = {'subsample': 0.5, 'n_estimators': 116, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 29, 'loss': 'quantile', 'learning_rate': 0.1}

            best_score : 数据类型为'float'
                模型交叉验证得分

        Examples
        --------
            gbrt, gbrt, gbrt_score = regr.gradientBoostRegression(X_train, y_train, 1, 200, 1, 50, 1, 2, 2, 3, [0.001, 0.01, 0.1, 1.0], [0.5, 0.6, 0.7, 0.8])
            # 直接使用predict()函数进行预测
            y_pre = gbrt.predict(X_test)
            # 使用utils.py中的saveModel()函数将模型保存到本地
            saveModel('gbrt.pkl', gbrt)
            # 使用utils.py中的loadModel()函数来调用本地模型，并进行预测
            y_pre = loadModel('gbrt.pkl').predict(X_test)                

        Notes
        --------
            自动调参默认开启全部CPU，如果需要更改，请修改源程序中gradientBoostingParm函数中的n_jobs参数，当n_jobs = -1时启用全部cpu

        '''
        from sklearn import ensemble
        best_parm, best_score = gradientBoostingParm(X_train, y_train,
                                                     n_estimators_start, n_estimators_end,
                                                     max_depth_start, max_depth_end,
                                                     min_samples_leaf_start, min_samples_leaf_end,
                                                     min_samples_split_start, min_samples_split_end,
                                                     learning_rate, subsample,
                                                     max_depth_step, n_estimators_step, min_samples_leaf_step,
                                                     min_samples_split_step,
                                                     cv_num=cv_num,
                                                     n_iter=n_iter,
                                                     n_jobs=-1)
        model = ensemble.GradientBoostingRegressor(n_estimators=best_parm['n_estimators'],
                                                   max_depth=best_parm['max_depth'],
                                                   loss=best_parm['loss'],
                                                   subsample=best_parm['subsample'],
                                                   learning_rate=best_parm['learning_rate'],
                                                   min_samples_leaf=best_parm['min_samples_leaf'],
                                                   min_samples_split=best_parm['min_samples_split'],
                                                   max_features=best_parm['max_features']).fit(X_train, y_train)
        return model, best_parm, best_score

    def lassoRegression(self, X_train, y_train, alpha, max_iter, cv_num=3, n_iter=20):
        '''
        lasso回归
        本函数需要设定的参数为alpha值和最大迭代次数

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
            
            alpha : 数据类型为'类数组'(list or numpy.narray)
                正则项系数。
                可以给定一个确定的量，例如alpha = [1.0]。
                如果不确定的话，也可以给定一个'类数组'，其中包含多个可能的取值，程序会自动选取最优值。例如alpha = [0.001, 0.01, 0.1, 1.0]。
                可以从10的幂次序列开始训练，根据效果微调。当 alpha 为 0 时算法等同于普通最小二乘法。
            
            max_iter : 数据类型为'int'
                最大迭代次数。
                次数越多越精准。

            cv_num : 数据类型为'int'
                S折交叉验证的折数。默认cv_num = 3
                即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

            n_iter : 数据类型为'int'
                训练次数，次数越大精度越高。默认n_iter=20

        Returns
        -------
            model : 数据类型为'object'
                训练好的模型。
                可以调用model.predict()函数可以对新的数据进行预测，具体使用方法可以见下方Examples
                可以使用utils.py中的saveModel函数将模型以.pkl的文件保存到本地。

            best_parm : 数据类型为'字典'
                模型最优参数。key为参数名称，value为对应值。
                例如best_parm = {'alpha': 0.1}

            best_score : 数据类型为'float'
                模型交叉验证得分

        Examples
        --------
            lasso, lasso_parm, lasso_score = regr.lassoRegression(X_train, y_train, 1, 200, 1, 50, 1, 2, 2, 3, [0.001, 0.01, 0.1, 1.0], [0.5, 0.6, 0.7, 0.8])
            # 直接使用predict()函数进行预测
            y_pre = lasso.predict(X_test)
            # 使用utils.py中的saveModel()函数将模型保存到本地
            saveModel('lasso.pkl', lasso)
            # 使用utils.py中的loadModel()函数来调用本地模型，并进行预测
            y_pre = loadModel('lasso.pkl').predict(X_test)                

        Notes
        --------
            自动调参默认开启全部CPU，如果需要更改，请修改源程序中lassoParm函数中的n_jobs参数，当n_jobs = -1时启用全部cpu

        '''
        from sklearn.linear_model import Lasso
        best_parm, best_score = lassoParm(X_train, y_train, alpha, max_iter, cv_num=cv_num, n_iter=n_iter, n_jobs=-1)
        model = Lasso(alpha=best_parm['alpha'], max_iter=max_iter).fit(X_train, y_train)
        return model, best_parm, best_score

    def nonlinearModelFitting(self, X_train, y_train, f, *parm, maxfev=20000):
        '''
        非线性最小二乘法拟合
        二维拟合曲线，三维拟合曲面，以此类推
        目前function.py中只有一元函数模型，如果需求需要自己添加，格式仿照function.py中其他函数格式即可。

        Parameters
        ----------
            X_train : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
                训练数据。
                数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
                当传入的f为一元函数时候X_train = [5.63100,6.80000,5.81800,6.80000,5.81800,5.42700,5.81800,5.42700,6.12900]
                当传入的f为多元函数时候X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

            y_train : 数据类型为'类数组'(list or numpy.narray)
                目标值(真实值)。
                每一个参数是对应样本数据的目标值
                例如y_train = [16.50000,31.10000,10.50000]
            
            f : 数据类型为float
                常用非线性模型函数在function.py中已经给出,使用的时候只需要写函数名，不要带括号

            parm : 数据类型为'类数组'(list or numpy.narray)
                初始参数列表
                
            maxfev : 数据类型为'int'
                最大拟合次数
                默认maxfev = 20000

        Returns
        -------
            res : 数据类型为'类数组'(list or numpy.narray)
                参数的最佳值。
                以使的平方残差之和最小，np.power(f(xdata, *res) - ydata, 2)
                
        '''
        from scipy.optimize import curve_fit
        res, pcov = curve_fit(
            f=f,
            xdata=X_train,
            ydata=y_train,
            p0=parm,
            maxfev=maxfev
        )
        return res
