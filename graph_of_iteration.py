import matplotlib.pyplot as plt
import numpy as np

nurd_width = 1000
x_coord_positions = np.linspace(1, nurd_width, nurd_width)
# x_new_coord_positions = np.linspace(1,nurd_width, nurd_width)
area = np.ones_like(x_coord_positions)
area2 = np.ones_like(x_coord_positions)
print(area)
print(x_coord_positions)

region = 0.4 # middle 0.6 to be rescaled per operations. 
scale = 0.92 # what scale is used to downsize per operation. 
total_length = area.shape[0]
print(total_length)
for i in range(10):
    left = int(0.5*total_length-0.5*region*total_length)
    # right = int(0.5*total_length+0.5*region*total_length)
    area[left:nurd_width-left] = area[left:nurd_width-left]*scale
    total_length = ((1-region)+(region*scale))*total_length
    print("length becomes : ", total_length)

x_new_coord_positions = np.cumsum(area)
print(x_new_coord_positions[-1])
plt.plot(x_coord_positions, x_new_coord_positions)
plt.plot(x_coord_positions[-1], x_new_coord_positions[-1], marker="o", markersize=10, markeredgecolor="black", markerfacecolor="red")
plt.text(x_coord_positions[-1]-140, x_new_coord_positions[-1], "(1000, 722.36)", ha='center', va='center')
plt.xlabel("Original x coordinates")
plt.ylabel("New rescaled x coordinates")
plt.title("x coordinates mapping from original to new frame \n(n=10, s=0.92, p=0.4)")
# plt.show()
plt.savefig("./scale_mapping2.png")

# plt.plot(x_coord_positions, area2)
# plt.plot(x_coord_positions, area)
# plt.fill_between(x_coord_positions, area2, color='orange', alpha=0.5)
# plt.fill_between(x_coord_positions, area, color='lightblue', alpha=1)
# plt.text(500, 0.7, "Down-sampled information", ha='center', va='center')
# plt.text(500, 0.05, "Total area of the rescaled NURD", ha='center', va='center')
# plt.xlabel("x units of the NURD region")
# plt.ylim(0,1.1)
# plt.ylabel("Effective scale across NURD region")
# plt.title("Iteration = 25, scale = 0.92, portion = 0.4")
# # plt.show()
# plt.savefig("./scale3.png")