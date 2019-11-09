'''
  File name: anms.py
  Author:
  Date created:
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed 
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''

def flatten(x,y,c):
	return x*c+y

def anms(cimg, max_pts):
	thres = 0.9
	w,h = cimg.shape
	min_radius = np.zeros((w*h,w*h))
	min_dist = []
	for xi in range(w):
		for yi in range(h):
			for xj in range(w):
				for yj in range(h):
					if (xi,yi) != (xj,yj) and cimg[xi,yi] < thres * cimg[xj,yj]:
						dist = distance.euclidean((xi,yi),(xj,yj))
						pi = flatten(xi,yi)
						pj = flatten(xj,yj)
						if dist < min_radius[pi,pj] or min_radius[pi,pj] == 0:
							min_radius[pi,pj] = dist
							min_dist.append((xi,yi,dist))
	min_dist.sort(key=lambda x:x[2], reverse=True)
	top_pts = min_dist[:max_pts]
	x = [x for (x,_,_) in top_pts]
	y = [y for (_,y,_) in top_pts]
	rmax = [r for (_,_,r) in top_pts]
	print(x)
	print(y)
	print(rmax)
	return x,y,rmax

# # test
# cimg = np.random.rand(5,10)
# anms(cimg, 8)

# # test
# I = np.array(Image.open('left.jpg').convert('RGB'))
# cimg = corner_detector(I)
# print(cimg.shape)
# x,y,rmax = anms(cimg, 50)

# plt.imshow('left.jpg')
# plt.scatter(x,y)
# plt.show()
