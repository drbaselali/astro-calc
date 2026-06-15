from scipy.spatial import distance
from sklearn.cluster import DBSCAN
import hdbscan
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import OPTICS
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.metrics.cluster import fowlkes_mallows_score 
import astropy
from astropy import units as u 

x , y, z = np.loadtxt('SAM.csv', unpack = True)
x2 = (x*u.pc*u.km/u.s).to('mpc^2/yr').value 
y2 = (y*u.pc*u.km/u.s).to('mpc^2/yr').value 
z2 = (z*u.pc*u.km/u.s).to('mpc^2/yr').value 
#x , y, z = np.loadtxt('SL32.txt', unpack = True)
#x , y, z = np.loadtxt('SLLL.txt', unpack = True)
x1 , y1, z1 = np.loadtxt('SLLL2.txt', unpack = True)

#oe = np.loadtxt('alloe.txt', unpack = True)
#x2, y2 = oe[4], oe[2]
#alloe= np.column_stack((x2 , y2))
#x , y, z = np.loadtxt('xyz.txt', unpack = True)
data = np.column_stack((x2 , y2, z2))
data = data.astype('float64')
uniform = np.column_stack((x1 , y1, z1))
uniform = uniform.astype('float32')


for i in range(4,8):
    model = hdbscan.HDBSCAN(min_cluster_size=i, gen_min_span_tree=True, metric='l1',alpha=1.0,approx_min_span_tree=True,min_samples=None)
    
    sns.set_context('poster')
    sns.set_style('white')
    sns.set_color_codes()
    plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}
    model.fit(data)
    model.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=80, edge_linewidth=2)
    model.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    model.condensed_tree_.plot()
    model.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
    palette = sns.color_palette()
    cluster_colors = [sns.desaturate(palette[col], sat) 
                      if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                      zip(model.labels_, model.probabilities_)]
    fig = plt.figure(figsize=(5,5))
    ax= Axes3D(fig)
    ax.scatter(data.T[0], data.T[1],data.T[2], c=cluster_colors)
    #ax3.view_init(elev=-90, azim=0)#LOS
    #ax.view_init(elev=90, azim=-90)#LOS
    #ax.view_init(elev=0, azim=-90)#Blackfo 0,-90 , original 0,-100
    #ax.view_init(elev=0, azim=0)#redfo 0, 0 , original -25, 0 
    #ax.view_init(elev=-15, azim=-45)#face on 
    #ax.view_init(elev=15, azim=+45)#edge on 

    plt.show()
    

    
    n_clusters_ = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
    n_noise_ = list(model.labels_).count(-1)





    print(i)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    #print("number of cluster found: {}".format(len(set(model.labels_))))
    print('cluster for each point: ', model.labels_)
  
mink = partial(distance.minkowski, p=1)    


model = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True, metric=mink,min_samples=3, cluster_selection_method = 'eom' ,alpha=1.0,approx_min_span_tree=True)
sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
sns.set(rc={'figure.figsize':(8,8)})
sns.set(font_scale = 1)
plot_kwds = {'alpha' : 0.5, 's' : 20, 'linewidths':0}
model.fit(data)
#plt.figure()
#model.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=80, edge_linewidth=2)
    
#model.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
#model.condensed_tree_.plot()
#plt.figure()
#model.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
palette = sns.color_palette()
cluster_colors = [sns.desaturate(palette[col], sat) if col >= 0 else (0.5, 0.5, 0.5) for col, sat in zip(model.labels_, model.probabilities_)]
#plt.figure()
#model.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    
fig=plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')
ax.set_aspect('equal')
ax.set_xlim(-30, 30)
ax.set_ylim(30, -30)
ax.set_zlim(-30, 30)
ax.scatter(data.T[0], data.T[1],data.T[2], c=cluster_colors, s = 30)
#ax.scatter(alloe.T[0], alloe.T[1], c=cluster_colors, s = 30)
ax.set_xlabel('h$_x$ [mpc$^2$/yr]', fontsize=10, fontweight='bold')
ax.set_ylabel('h$_y$ [mpc$^2$/yr]', fontsize=10, fontweight='bold')
ax.set_zlabel('h$_z$ [mpc$^2$/yr]', fontsize=10, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=10)
ax.grid()
ax.view_init(elev=-90, azim=0)#Blackfo 0,-90 , original 0,-100
#ax.view_init(elev=0, azim=0)#redfo 0, 0 , original -25, 0 
#ax.view_init(elev=-15, azim=-45)#face on 
#ax.view_init(elev=15, azim=+45)#edge on 
#ax.view_init(elev=-90, azim=90)#green edge-on
#fig.savefig('poleclustering.pdf') 
#plt.clf()
n_clusters_ = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
n_noise_ = list(model.labels_).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
#     #print("number of cluster found: {}".format(len(set(model.labels_))))
print('cluster for each point: ', model.labels_)
print(model.probabilities_) 
print(model.outlier_scores_)
#print(model.cluster_persistence_)
#print(model.relative_validity_)
print(hdbscan.validity.validity_index(data, model.labels_, metric = distance.cosine, per_cluster_scores = True, d = 3))

        

