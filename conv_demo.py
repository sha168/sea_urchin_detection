import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.animation

#####################
# Array preparation
#####################

#input array
a = np.random.randint(50,150, size=(5,5))
a = np.zeros((10, 10))
a[:, 5:] = 1
# kernel
kernel = np.array([[ -1,0, 1],
                   [-2, 0,2],
                   [ -1, 0, 1]])

# visualization array (2 bigger in each direction)
va = np.ones((a.shape[0], a.shape[1]), dtype=int) * a
#va[1:-1,1:-1] = a

#output array
res = np.zeros_like(a[:-2,:-2])

#colorarray
va_color = np.ones((a.shape[0], a.shape[1])) * a * 0.5
#va_color[1:-1,1:-1] = 0.5

#####################
# Create inital plot
#####################
fig = plt.figure(figsize=(8,4))

def add_axes_inches(fig, rect):
    w,h = fig.get_size_inches()
    return fig.add_axes([rect[0]/w, rect[1]/h, rect[2]/w, rect[3]/h])

axwidth = 3.
cellsize = axwidth/va.shape[1]
axheight = cellsize*va.shape[0]

ax_va  = add_axes_inches(fig, [cellsize, cellsize, axwidth, axheight])
ax_kernel  = add_axes_inches(fig, [cellsize*2+axwidth,
                                   (res.shape[0])*cellsize-kernel.shape[0]*cellsize,
                                   kernel.shape[1]*cellsize,
                                   kernel.shape[0]*cellsize])
ax_res = add_axes_inches(fig, [cellsize*3+axwidth+kernel.shape[1]*cellsize,
                               2*cellsize,
                               res.shape[1]*cellsize,
                               res.shape[0]*cellsize])
ax_kernel.set_title("Kernel", size=12)

im_va = ax_va.imshow(va_color, vmin=0., vmax=1.3, cmap="Blues")
for i in range(va.shape[0]):
    for j in range(va.shape[1]):
        ax_va.text(j,i, va[i,j], va="center", ha="center")

ax_kernel.imshow(np.zeros_like(kernel), vmin=-1, vmax=1, cmap="Pastel1")
for i in range(kernel.shape[0]):
    for j in range(kernel.shape[1]):
        ax_kernel.text(j,i, kernel[i,j], va="center", ha="center")


im_res = ax_res.imshow(res, vmin=0, vmax=1.3, cmap="Greens")
res_texts = []
for i in range(res.shape[0]):
    row = []
    for j in range(res.shape[1]):
        row.append(ax_res.text(j,i, "", va="center", ha="center"))
    res_texts.append(row)


for ax  in [ax_va, ax_kernel, ax_res]:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.yaxis.set_major_locator(mticker.IndexLocator(1,0))
    ax.xaxis.set_major_locator(mticker.IndexLocator(1,0))
    ax.grid(color="k")

###############
# Animation
###############
def init():
    for row in res_texts:
        for text in row:
            text.set_text("")

def animate(ij):
    i,j=ij
    o = kernel.shape[1]//2
    # calculate result
    res_ij = (kernel*va[1+i-o:1+i+o+1, 1+j-o:1+j+o+1]).sum()
    res_texts[i][j].set_text(res_ij)
    # make colors
    c = va_color.copy()
    c[1+i-o:1+i+o+1, 1+j-o:1+j+o+1] = 1.
    im_va.set_array(c)

    r = res.copy()
    r[i,j] = 1
    im_res.set_array(r)

i,j = np.indices(res.shape)
ani = matplotlib.animation.FuncAnimation(fig, animate, init_func=init,
                                         frames=zip(i.flat, j.flat), interval=400)
ani.save("algo.gif", writer="imagemagick")
plt.show()