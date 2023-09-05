import matplotlib.pyplot as plt
import numpy as np

# A trivial example of the overall scale factor for rescale operation 

def main(): 
    s= 0.9 
    data = np.ones(1000)
    x_data = np.linspace(0, 100, 1000)
    length = 1000
    x = [] 
    y = []
    for i in range(3): 
        x.append(0.2*length)
        y.append(1000-0.2*length)
        data[int(0.2*length):int(1000-0.2*length)] = data[int(0.2*length):int(1000-0.2*length)] * s
        length = 0.6*length*s + 0.4*length
        print(length)
        
    print("Final length = ", length)
    print(x)
    print(y)    
    ori = []
    new = [] 
    ori.append(0)
    ori.append(x[-1])
    ori.append(x[-2])
    ori.append(x[-3])
    ori.append(y[0])
    ori.append(y[1])
    ori.append(y[2])
    ori.append(1000)
    new.append(0)
    new.append(x[-1])
    new.append((x[-2]-x[-1])*s+x[-1])
    new.append((x[-3]-new[-1])*s*s+new[-1])
    new.append(new[-1]+s*s*s*(600))
    new.append(new[-1]+12*s*s)
    new.append(new[-1]+11.28*s)
    new.append(1000-823.28+new[-1])
    # data[]
    # print(data, x_data)
    plt.plot(x_data, data)
    plt.ylim(0,1.1)
    plt.show()

    new = [ele/10 for ele in new]
    ori = [e/10 for e in ori]
    plt.plot(ori, new)
    plt.scatter(ori, new)
    plt.ylabel('new content consist of original points at x = ')
    plt.xlabel('original content at x = ')
    plt.title('Content re-distribution of a resize operation when n = 3')
    plt.show()

    new = [elements/max(new) for elements in new]
    ori = [elements/max(ori) for elements in ori]
    plt.plot(ori, new)    
    plt.ylabel('new content')
    plt.xlabel('original content')
    plt.title("normalized version")
    plt.show()

    


if __name__ == "__main__":
    main()