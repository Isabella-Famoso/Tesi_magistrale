import numpy as np
from scipy.interpolate import PPoly
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from plotbin.display_pixels import display_pixels 

#increment arrays
increments_composite = np.arange(0.62, 1.52, 0.05)   
increments_SF = np.arange(0.62, 1.12, 0.05)   
increments_LINER = np.arange(0.44, -0.24, -0.05) 

#classification curves
def kewley_curve_NII(x): return (0.61 / (x - 0.47)) + 1.19
def kauffmann_curve_NII(x): return (0.61 / (x - 0.05)) + 1.3

#shifted curves for each class
def CompositeNESS(x, i): return (i / (x - 0.47)) + 1.19
def SFNESS(x, i): return (i / (x - 0.05)) + 1.3
def LINERNESS(x, i): return 1.05 * x + i


#------------------------------------------NEW--------------------------------
def inverse_kewley(y):
    x=(0.051+0.47*y)/ (y-1.19)
    return x
    
def inverse_kauffmann(y):
    x= (0.05*y +0.545)/(y - 1.3)
    return x


#Newfunction
def classify_BPT(N2Ha,O3Hb):

	mask = np.zeros(N2Ha.shape[0])
	for i in range(N2Ha.shape[0]):
		x_kewley = inverse_kewley(O3Hb[i])
		x_kauffmann = inverse_kauffmann(O3Hb[i])
		if N2Ha[i] < x_kauffmann: 
			mask[i] = 1                 #1 is for SF
		elif N2Ha[i]> x_kauffmann and N2Ha[i]<x_kewley:
			mask[i] = 2                 #2 is for composite
		elif N2Ha[i]>x_kewley:
			mask[i]=3                   #3 is for AGN
		else:
			mask[i]=np.nan
	mask[O3Hb>1]=3
	return mask
     
        	
def plotBPT(N2Ha,O3Hb):
	mask = classify_BPT(N2Ha,O3Hb)
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(1,1,1)
	ax.plot(N2Ha[mask==1],O3Hb[mask==1], '.b')
	ax.plot(N2Ha[mask==3],O3Hb[mask==3], '.g')
	ax.plot(N2Ha[mask==2],O3Hb[mask==2], '.y')
	x_kewley = np.linspace(-1.28, 0.45, 500)
	x_kauffmann = np.linspace(-1.5, -0.167, 500)	  
	ax.plot(x_kewley, kewley_curve_NII(x_kewley), 'black', label='Kewley (BPT)', linewidth=1.5)
	ax.plot(x_kauffmann, kauffmann_curve_NII(x_kauffmann), 'k--', label='Kauffmann (BPT)', linewidth=1.5)
	ax.set_ylim(-3,3)
	plt.show()

def computeX0(x1,y1):
	x = np.linspace(-3,2,1000)
	y = 40000*x**4 - 40000*x1*x**3 - 6000*x**3 + 6000*x**2*x1 + 300*x**2 - 300*x*x1 - 31725*x + 5*x1 * 24400*y1*x - 1220*y1 - 13298 
	
	plt.plot(x,y)
	plt.show()

	  
def compute_d1(x1,y1):
	x = np.linspace(-2,1.0,10000)
	distance = np.sqrt((x1-x)**2 + (y1 - (0.61/(x-0.05)+1.3))**2)
	x0 = x[np.argmin(distance)]
	d1 = np.min(distance)
	return d1
    
def compute_d2(x1,y1):
	x = np.linspace(-2,2,10000)
	distance = np.sqrt((x1-x)**2 + (y1 - (0.61/(x-0.47)+1.19))**2)
	x0 = x[np.argmin(distance)]
	d2 = np.min(distance)
	#plt.plot(x,y)
	#plt.show()
	return d2	
    
def compute_eta(N2Ha,O3Hb):
	mask = classify_BPT(N2Ha,O3Hb)
	eta = np.zeros(mask.shape[0])
	for i in range(mask.shape[0]):
		if mask[i]==1:
			d1=compute_d1(N2Ha[i],O3Hb[i])
			eta[i]= (-0.5-d1)
		elif mask[i]==2:
			d1=compute_d1(N2Ha[i],O3Hb[i])
			d2=compute_d2(N2Ha[i],O3Hb[i])
			eta[i]= (-0.5 + d1/(d1+d2))
		elif mask[i]==3:
			d2=compute_d2(N2Ha[i],O3Hb[i])
			eta[i]=0.5+d2
		else:
			eta[i]=np.nan
	return eta
#####
def plotBPT_gradient(X, Y, N2Ha, O3Hb, grey_mask=None):
	eta = compute_eta(N2Ha, O3Hb)

	# ---------- BPT 2D diagram ----------
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(1,1,1)

	valid = ~np.isnan(eta)
	scat = ax.scatter(N2Ha[valid], O3Hb[valid], c=eta[valid], cmap='jet', vmin=-1, vmax=1, marker='o', edgecolors='black', alpha=1, linewidths=0.4)

	if grey_mask is not None:
		good = ~np.isnan(N2Ha) & ~np.isnan(O3Hb)
		grey_spaxels = grey_mask & good
		ax.scatter(N2Ha[grey_spaxels], O3Hb[grey_spaxels], facecolors='none',edgecolors='grey', alpha=1, marker='o')

	cbar = plt.colorbar(scat, ax=ax)
	cbar_ax = cbar.ax
	cbar_ax.text(0.4, 0.89, r'AGN', rotation=90, fontsize=15, color='white', ha='left', va='center', transform=cbar_ax.transAxes)
	cbar_ax.text(0.4, 0.1, r'SF', rotation=90, fontsize=15, color='white', ha='left', va='center', transform=cbar_ax.transAxes)

	x_kewley = np.linspace(-1.28, 0.45, 500)
	x_kauffmann = np.linspace(-1.5, -0.167, 500)
	ax.plot(x_kewley, kewley_curve_NII(x_kewley), 'black', label='Kewley (BPT)', linewidth=2)
	ax.plot(x_kauffmann, kauffmann_curve_NII(x_kauffmann), 'k--', label='Kauffmann (BPT)', linewidth=2)

	ax.set_ylim(-1.5, 1.5)
	ax.set_xlim(-1.5, 0.75)
	ax.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 0.75])
	ax.set_xlabel(r'$\log([\mathrm{NII}]/\mathrm{H}\alpha)$', fontsize=15)
	ax.set_ylabel(r'$\log([\mathrm{OIII}]/\mathrm{H}\beta)$', fontsize=15)
	plt.title('[NII] standard 2D BPT', fontsize=15)
	fig.show()

	# ---------- Spatial BPT Map ----------
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(1,1,1)

	# Main coloredspaxels
	scat = display_pixels(X, Y, eta, pixelsize=1.2, angle=0, cmap='jet', vmin=-1, vmax=1, alpha=1)
	plt.colorbar(scat, ax=ax)

	#put grey spaxels where grey_mask is True
	if grey_mask is not None:
		valid_grey = grey_mask & ~np.isnan(eta)
		ax.scatter(X[valid_grey], Y[valid_grey], s=40, marker='s',facecolors='lightgrey', edgecolors='lightgrey', alpha=1)

	plt.xlabel('X (pixels)', fontsize=15)
	plt.ylabel('Y (pixels)', fontsize=15)
	plt.title('[NII] spatial BPT', fontsize=15)
	fig.show()

##q
