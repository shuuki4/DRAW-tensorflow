import cPickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

with_filters = True # if results were generated 
file_name = 'generated_images_filters_100.npy'

with open(file_name, 'rb') as f :
	if with_filters : images, filters = cPickle.load(f)
	else : images = cPickle.load(f)

plt.figure(figsize=(8,8))
plt.gray()
gs = gridspec.GridSpec(8, 8)
gs.update(wspace=0.00, hspace=0.00)
n = 5.0

for t in range(images.shape[0]) :
	
	if t>0 and with_filters :
		mu_x, mu_y, delta, var = filters[t-1]

	plt.clf()
	for i in range(64) :
		ax = plt.subplot(gs[i])
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		ax.imshow(np.squeeze(images[t, i, :, :, :]))
		if t>0 and with_filters :
			ax.add_patch(patches.Rectangle((mu_x[i]-1-(n/2)*delta[i], mu_y[i]-1-(n/2)*delta[i]), n*delta[i], n*delta[i], fill=False, edgecolor='red', linewidth=var[i] ))
	plt.savefig('attention_image_'+str(t)+'.png', format='png')