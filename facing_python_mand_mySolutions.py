'''
------------------- 
Mandatory exercises
-------------------
'''

import numpy as np
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
import matplotlib.image as img
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
plt.ion()
''' Read data from the image 'facing_python.jpg' and store it in the
numpy array 'rgb_data'. The image measures 600x441 pixels. The color
of each pixel is given by a RGB-triplet. These three values define the
amount of red, green and blue, respectively, and can range between 0
and 255. '''
rgb_data = img.imread('facing_python.jpg')

''' Now plot the image (data) with the help of the function imshow(). '''
fig1 = plt.figure('Original image')
plt.imshow(rgb_data/255.,origin='lower')
plt.axis('off')

''' (1) Check the shape of the array 'rgb_data'. '''

print rgb_data.shape

''' (2) reshape() the 3D-array 'rgb_data' to construct a 2D-array
'flat_data' containing just the RGB-triplets '''

flat_data = rgb_data.reshape((rgb_data.shape[0]*rgb_data.shape[1]),rgb_data.shape[-1])
# flat_data = rgb_data.reshape(np.prod(rgb_data.shape[:2]),rgb_data.shape[-1])
print flat_data.shape

''' (3) Make sure that in any of the following plots axis labels have
font size 14. '''

mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['figure.facecolor'] = 'w'

''' (4) Extract and plot the principal colour values '''

w = 4
h = 5
fig2 = plt.figure('Histograms', figsize=(h, w))
fig2.subplots_adjust(left = 0.16,bottom=0.15) # to make sure we see the axis labels
fig2.suptitle('Distribution of Colour Values')
colors = ['red','green','blue']
mybins = np.arange(0.,256.,1.)
for ii in range(3):
    ax = fig2.add_subplot(3,1,ii+1)
    ax.hist(flat_data[:,ii],bins=mybins,fc=colors[ii],ec=colors[ii])
    
    ax.set_xlim([flat_data.min(),flat_data.max()])
    ax.set_ylim([0.,6000.])
    ax.set_yticks(np.arange(0,6001.,2000.))
    if ii==1:  # this is the middle plot, so we want to have the y-label, to plot text at any position look at plt.figtext()
        ax.set_ylabel(r'$N_\mathrm{pix}$') # we need the r for latex style
        
    if ii==2:
        ax.set_xlabel('color value') # we want to plot the xlabel at the lowest of the three plots
    else:
        ax.set_xticklabels([]) # we remove the xticklabels; other solutions could be e.g. ax.axes.get_xaxis().set_visible(False)
fig2.savefig('rgb_histograms.png')

''' (5) Scatter plot the rgb data '''

fig3 = plt.figure('First scatter plot')
ax = Axes3D(fig3)
ax.set_title('Scatter-Plot of RGB Data')
ax.scatter(flat_data[::100,0],flat_data[::100,1],flat_data[::100,2],c=flat_data[::100]/255., edgecolor = flat_data[::100]/255.)
# it is important that you use keyword c instead of color. Otherwise you end up in blue dots
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
ax.set_xlim3d([0,255])
ax.set_ylim3d([0,255])
ax.set_zlim3d([0,255])
plt.savefig('rgb_3d_plot.png')



''' (6) Cluster the rgb data using kmeans2 '''
nclust = 5
flat_data = np.array(flat_data, dtype=float) # we cast from integers to floats, you can also just devide by 1.0
centroids,labels = kmeans2(flat_data,nclust)


''' (7) Replot the clustered data '''

fig4 = plt.figure('Reclustered scatter')
ax = Axes3D(fig4)
ax.set_title('Color Reduction by Clustering')
ax.scatter(flat_data[::100,0],flat_data[::100,1],flat_data[::100,2],c=centroids[labels[::100]]/255., edgecolor=centroids[labels[::100]]/255.)
# centroid has the rgb values, labels indicates wich cluster we have and is therefore the index of the centroid

ax.plot(centroids[:,0],centroids[:,1],centroids[:,2],'k*',markersize=10)
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
ax.set_xlim3d([0,255])
ax.set_ylim3d([0,255])
ax.set_zlim3d([0,255])
plt.savefig('kmeans_3d.png')


''' (8) Replot the image using the reduced dimensionality rgb triplets '''
reduced_data = centroids[labels]/255.
reduced_image = reduced_data.reshape(rgb_data.shape[0],rgb_data.shape[1],3)
fig5 = plt.figure('Last Figure')
plt.imshow(reduced_image,origin='lower')
plt.axis('off')
plt.savefig('five_c_img.png')

