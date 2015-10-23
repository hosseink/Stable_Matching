import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

def addconnection(i,j,c):
  return [((-1,1),(i-1,j-1),c)]

def drawnodes(s,i, w, h):
  global ax
  if(i==1):
    color='r'
    posx=1
  else:
    color='b'
    posx=-1

  posy=0
  for n in s:
    plt.gca().add_patch( plt.Circle((posx,posy), .1,fc=color))
    if posx==1:
      ax.annotate(n,xy=(posx,posy+0.1))
    else:
      ax.annotate(n,xy=(posx-len(n)*0.1,posy+0.1))
    posy+=1

set1=['Man1','Man2','Man3','Man4']
set1 = ['man' for _ in range(50)]
set2=['Woman1','Woman2','Woman3','Woman4','Woman5']
ax=plt.figure(figsize = (4, max(len(set1), len(set2)))).add_subplot(111)
plt.axis([-2,2,-1,max(len(set1),len(set2))+1])
#plt.axis("equal")
frame=plt.gca()
frame.axes.get_xaxis().set_ticks([])
frame.axes.get_yaxis().set_ticks([])

x0, y0 = ax.transAxes.transform((0,0))
x1, y1 = ax.transAxes.transform((1,1))
dx = x1 - x0
dy = y1 - y0
maxd = max(dx, dy)
w = maxd/ dx
h = maxd/ dy
print w, h

drawnodes(set1,1, w, h)
drawnodes(set2,2, w, h)

connections=[]
connections+=addconnection(1,2,'g')
connections+=addconnection(1,3,'y')
connections+=addconnection(1,4,'g')
connections+=addconnection(2,1,'g')
connections+=addconnection(4,1,'y')
connections+=addconnection(4,3,'g')
connections+=addconnection(5,4,'y')

for c in connections:
  plt.plot(c[0],c[1],c[2])

plt.show()
