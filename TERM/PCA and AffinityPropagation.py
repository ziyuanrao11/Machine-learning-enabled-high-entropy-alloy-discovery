# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 15:44:05 2022

@author: z.rao
"""


from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
def plot_gmm(gm, principalDf, label=True, ax=None):
    fig, axs = plt.subplots(1,1)
    plt.scatter(c_trans[:,0],c_trans[:,1], s = 10, c='black', marker='^',zorder=2)
    ax = axs or plt.gca()
    # labels = gm.fit(X).predict(X)
    ax.scatter(principalDf[:, 0], principalDf[:, 1], s=5, zorder=2)
    # plt.axis('equal')
    # plt.xlim(-0.16,0.27)
    # plt.ylim(-0.023,0.035)
    w_factor = 0.2 / gm.weights_.max()
    for pos, covar, w in zip(gm.means_, gm.covariances_, gm.weights_):
        draw_ellipse(pos, covar, alpha= 0.85*w * w_factor, facecolor='slategrey', zorder=-10)
    #fig.savefig('PC_GMM_1.tif', bbox_inches = 'tight', pad_inches=0.01)
def APropagation(X):
    # fig = plt.figure(figsize = (12,8))
    model = AffinityPropagation(damping=0.9)
    # 匹配模型
    model.fit(X)
    # 为每个示例分配一个集群
    yhat = model.predict(X)
    # 检索唯一群集
    clusters = unique(yhat)
    # 为每个群集的样本创建散点图
    out = np.empty([1,2])
    # plt.figure(figsize=(12,8),dpi=200)
    for cluster in clusters:
    # 获取此群集的示例的行索引
        row_ix = where(yhat == cluster)
        # 创建这些样本的散布
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1],zorder=1)
        ind = np.arange(len(X[row_ix]))
        sub_ind = np.random.choice(ind, 1, replace=False)
        a= np.array(X[row_ix])[sub_ind]
        pyplot.scatter(a[0, 0], a[0, 1], s=15, c='black', marker='^',zorder=2)
        out=np.append(out,a,axis=0)
        # Out=Out.append(a, axis=1)
    pyplot.savefig('PC_AP_1.tif', bbox_inches = 'tight', pad_inches=0.01)
    return out
principalDf = principalDf.to_numpy()
gm = GaussianMixture(n_components=4, random_state=0, init_params='kmeans').fit(principalDf)
print('Average negative log likelihood:', -1*gm.score(principalDf))
plot_gmm(gm,principalDf)
out = APropagation(principalDf)
out = pca.inverse_transform(out)
out=pd.DataFrame(out,columns= ['Fe','Ni','Co','Cr','V','Cu'])
out=out.drop(labels=0)
out=out.round(3)
out.to_csv('DFT_C_1.csv')
