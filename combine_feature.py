import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import math  

# An attempt to produce the correlation of % difference in standard width 
# of an OCT image and the feature matching score with its template. 
# Not very effective..

def main() : 
    path = "./feature_matching/"
    alpha_ex = 0.01 # 0.01orb 0.2tm
    match = cv2.imread(path +"angular_scanning/90-360/90 copy.png")
    match = cv2.resize(match, (int(match.shape[1]*0.6), match.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    print("sub-region shape = ", match.shape)
    template = cv2.imread(path+"new/rescale_template.png")
    print("Template shape = ", template.shape)
    match = cv2.cvtColor(match, cv2.COLOR_BGR2GRAY)
    # match = cv2.equalizeHist(match)
    # template = cv2.medianBlur(template, 5)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    x = []
    y = []
    for i in range(300):
        scale = 0.85 + 0.001*i
        print("scale factor = ", scale)
        matching = match
        # matching = cv2.resize(matching, (int(matching.shape[1]*scale*portion), template.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        matching = cv2.resize(matching, (int(matching.shape[1]*scale), template.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        print("matching shape = ", matching.shape)
        print("template shape = ", template.shape)
        print(matching.shape, template.shape)

        # ==============================================================================
        # result = cv2.matchTemplate(matching, template, cv2.TM_SQDIFF_NORMED)
        # result = cv2.matchTemplate(matching, template, cv2.TM_CCOEFF_NORMED)
        # max_similarity = np.max(result)
        # ==============================================================================
        sift = cv2.ORB_create()
        # Detect keypoints and compute descriptors for the smaller image
        keypoints_object, descriptors_small = sift.detectAndCompute(matching, None)
        # Detect keypoints and compute descriptors for the larger image
        keypoints_template, descriptors_large = sift.detectAndCompute(template, None)
        # Create a BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(descriptors_small, descriptors_large, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.90 * n.distance:
                good_matches.append(m)
        max_similarity = len(good_matches) / len(matches)
        # ==============================================================================

        print("score = ", max_similarity)
        # print("Normalized score = ", normalized_score)
        x.append(scale-1)
        y.append(max_similarity)

    print("scale and score = ", x , y)
    # Set the window size for the moving average
    df1 = pd.DataFrame({
        'x': x,
        'y': y
    })
    df1['EMA1'] = df1['y'].ewm(alpha=alpha_ex, adjust=False).mean()
    # window_size = 5
    # # Calculate the moving average for x and y data
    # x_ma1 = np.convolve(x, np.ones(window_size), 'valid') / window_size
    # y_ma1 = np.convolve(y, np.ones(window_size), 'valid') / window_size

    
    match2 = cv2.imread(path +"angular_scanning/180-360/180 copy.png")
    match2 = match2[:,500:2581,:]
    match2 = cv2.resize(match2, (int(match2.shape[1]*1.05), match2.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    # print("sub-region shape = ", match.shape)
    # template2  = cv2.imread(path+"ref_images/template.png")
    template = cv2.imread(path+"new/rescale_template.png")
    # print("Template shape = ", template.shape)
    match2 = cv2.cvtColor(match2, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    x = []
    y = []

    for i in range(300):
        scale = 0.85 + 0.001*i
        print("scale factor = ", scale)
        matching = match2
        # matching = cv2.resize(matching, (int(matching.shape[1]*scale*portion), template.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        matching = cv2.resize(matching, (int(matching.shape[1]*scale), template.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        print("matching shape = ", matching.shape)
        print("template shape = ", template.shape)

        # ==============================================================================
        # result = cv2.matchTemplate(matching, template, cv2.TM_SQDIFF_NORMED)
        # result = cv2.matchTemplate(matching, template, cv2.TM_CCOEFF_NORMED)
        # max_similarity = np.max(result)
        # ==============================================================================
        sift = cv2.ORB_create()
        # Detect keypoints and compute descriptors for the smaller image
        keypoints_object, descriptors_small = sift.detectAndCompute(matching, None)
        # Detect keypoints and compute descriptors for the larger image
        keypoints_template, descriptors_large = sift.detectAndCompute(template, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(descriptors_small, descriptors_large, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.90 * n.distance:
                good_matches.append(m)
        max_similarity = len(good_matches) / len(matches)
        # ==============================================================================
        print("score = ", max_similarity)
        # print("Normalized score = ", normalized_score)
        x.append(scale-1)
        y.append(max_similarity)

    print("scale and score = ", x , y)
    # Set the window size for the moving average
    df2 = pd.DataFrame({
        'x': x,
        'y': y
    })
    df2['EMA2'] = df2['y'].ewm(alpha=alpha_ex, adjust=False).mean()
    # window_size = 5
    # # Calculate the moving average for x and y data
    # x_ma2 = np.convolve(x, np.ones(window_size), 'valid') / window_size
    # y_ma2 = np.convolve(y, np.ones(window_size), 'valid') / window_size

    match3 = cv2.imread(path +"angular_scanning/180-360/180 copy.png")
    match3 = cv2.resize(match3, (int(match3.shape[1]*1.1), match3.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    # print("sub-region shape = ", match.shape)
    # template2  = cv2.imread(path+"ref_images/template.png")
    template = cv2.imread(path+"new/rescale_template.png")
    # print("Template shape = ", template.shape)
    match3 = cv2.cvtColor(match3, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    x = []
    y = []

    for i in range(300):
        scale = 0.85 + 0.001*i
        print("scale factor = ", scale)
        matching = match3
        # matching = cv2.resize(matching, (int(matching.shape[1]*scale*portion), template.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        matching = cv2.resize(matching, (int(matching.shape[1]*scale), template.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        print("matching shape = ", matching.shape)
        print("template shape = ", template.shape)

        # ==============================================================================
        # result = cv2.matchTemplate(matching, template, cv2.TM_SQDIFF_NORMED)
        # result = cv2.matchTemplate(matching, template, cv2.TM_CCOEFF_NORMED)
        # max_similarity = np.max(result)
        # ==============================================================================
        sift = cv2.ORB_create()
        # Detect keypoints and compute descriptors for the smaller image
        keypoints_object, descriptors_small = sift.detectAndCompute(matching, None)
        # Detect keypoints and compute descriptors for the larger image
        keypoints_template, descriptors_large = sift.detectAndCompute(template, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(descriptors_small, descriptors_large, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.90 * n.distance:
                good_matches.append(m)
        max_similarity = len(good_matches) / len(matches)
        # ==============================================================================
        print("score = ", max_similarity)
        # print("Normalized score = ", normalized_score)
        x.append(scale-1)
        y.append(max_similarity)

    print("scale and score = ", x , y)
    # Set the window size for the moving average
    df3 = pd.DataFrame({
        'x': x,
        'y': y
    })
    df3['EMA3'] = df3['y'].ewm(alpha=alpha_ex, adjust=False).mean()
    # window_size = 5
    # # Calculate the moving average for x and y data
    # x_ma3 = np.convolve(x, np.ones(window_size), 'valid') / window_size
    # y_ma3 = np.convolve(y, np.ones(window_size), 'valid') / window_size

    # plt.plot(x, y)
    plt.plot(df1['x'], df1['EMA1'], label = 'angular_90', color='b')
    plt.plot(df2['x'], df2['EMA2'], label = 'angular_120', color='r')
    plt.plot(df3['x'], df3['EMA3'], label = 'angular_180', color='g')
    plt.fill_between(df1['x'], df1['EMA1']-0.015 * df1['EMA1'], df1['EMA1']+ 0.015 * df1['EMA1'], alpha=0.2, color='b')
    plt.fill_between(df2['x'], df2['EMA2']-0.012 * df2['EMA2'], df2['EMA2']+ 0.012 * df2['EMA2'], alpha=0.2, color='r')
    plt.fill_between(df3['x'], df3['EMA3']-0.01 * df3['EMA3'] , df3['EMA3']+0.01 * df3['EMA3'], alpha=0.2, color='g')
    plt.xlabel("scale length (ratio) ")
    plt.ylabel("Correlation with the template")
    # plt.show()
    plt.legend()
    plt.savefig("./feature_match_combine.png")


if __name__ == "__main__":
    main()