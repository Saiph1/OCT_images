import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import math  

# An attempt to produce the correlation of % difference in standard width 
# of an OCT image and the feature matching score with its template. 
# Not very effective..

def main() : 
    path = "../../test_dis/feature_matching/"
    match = cv2.imread(path +"angular_scanning/90-360/90 copy.png")
    # print("Original size = ", match.shape)
    # match = match[:, int(match.shape[1]*0.1):int(match.shape[1]*0.9)]
    # print("Tyr: ", match.shape)
    portion = 90/360
    # match2 = cv2.imread(path +"angular_scanning/90-360/90 copy.png")
    # match = cv2.imread(path +"new/90.oct.png")
    # match = cv2.cvtColor(match, cv2.COLOR_BGR2GRAY)
    print("sub-region shape = ", match.shape)
    # template2  = cv2.imread(path+"ref_images/template.png")
    template = cv2.imread(path+"new/rescale_template.png")
    # template = cv2.imread(path+"ref_images/template copy.png")
    # template = cv2.resize(template, (template2.shape[1], template2.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    print("Template shape = ", template.shape)
    # =========
    # l1 = 5567
    # l2 = 7783
    # l1 = 1209
    # l2 = 4856
    # l1 = 3730
    # l2 = 9191
    # =========
    # Testing whether the result will be different if the tamplate the match has the same height
    # template = cv2.resize(template, (template.shape[1], int(match.shape[0])), interpolation=cv2.INTER_LANCZOS4)
    # Enhance the computation by increasing the contrast and removing noise.
    # Remove salt and pepper noise using median filtering
    # match = cv2.medianBlur(match, 5)
    match = cv2.cvtColor(match, cv2.COLOR_BGR2GRAY)
    # match = cv2.equalizeHist(match)
    # template = cv2.medianBlur(template, 5)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # template = cv2.equalizeHist(template)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # match = clahe.apply(match)
    # template = clahe.apply(template)
    # cv2.imshow("match", match)
    # cv2.imshow("template", template)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    x = []
    y = []
    for i in range(500):
        scale = 0.75 + 0.001*i
        print("scale factor = ", scale)
        matching = match
        # matching = cv2.resize(matching, (int(matching.shape[1]*scale*portion), template.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        matching = cv2.resize(matching, (int(matching.shape[1]*scale), template.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        # if matching.shape[1] < 6244:
        #     white = np.ones((1024, 6244)) * 255
        #     white[:, :matching.shape[1]] = matching
        #     matching = white.astype(np.uint8)
        print("matching shape = ", matching.shape)
        print("template shape = ", template.shape)
        # ============================================================
        # result = cv2.matchTemplate(matching, template, cv2.TM_CCOEFF_NORMED)
        # max_similarity = np.max(result)
        print(matching.shape, template.shape)
        # result = cv2.matchTemplate(matching, template, cv2.TM_SQDIFF_NORMED)
        # max_similarity = np.min(result)
        # print(result)
        # ======================================================================
        # sift = cv2.xfeatures2d.SIFT_create()
        sift = cv2.ORB_create()
        # Detect keypoints and compute descriptors for the smaller image
        keypoints_object, descriptors_small = sift.detectAndCompute(matching, None)
        # Detect keypoints and compute descriptors for the larger image
        keypoints_template, descriptors_large = sift.detectAndCompute(template, None)
        # Create a BFMatcher object
        # ====================================
        # Initialize the feature matcher
        # matcher = cv2.BFMatcher()
        # # Match the descriptors of keypoints between the smaller and larger images
        # Match descriptors using k-NN algorithm
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(descriptors_small, descriptors_large, k=2)
        # Apply ratio test to filter good matches
        # ====================================
        # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # matches = matcher.knnMatch(descriptors_small, descriptors_large, k=2)
        # matcher = cv2.BFMatcher()
        # matches = matcher.match(descriptors_small, descriptors_large)
        # total_keypoints = max(len(keypoints_object), len(keypoints_template))
        # good_matches = [match for match in matches if match.distance < 0.7 * max(match.distance for match in matches)]
        # num_good_matches = len(good_matches)
        # max_similarity = num_good_matches / total_keypoints
        # ====================================
        good_matches = []
        for m, n in matches:
            if m.distance < 0.90 * n.distance:
                good_matches.append(m)
        max_similarity = len(good_matches) / len(matches)
        # ===================================================================
        
        # ===================================================================
        # Method 2:
        # Perform geometric verification using RANSAC
        # MIN_MATCH_COUNT = 5  # Minimum number of matches required for RANSAC
        # if len(good_matches) >= MIN_MATCH_COUNT:
        #     src_pts = np.float32([keypoints_small[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        #     dst_pts = np.float32([keypoints_large[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        #     # Estimate the transformation matrix using RANSAC
        #     M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        #     # Calculate the number of inlier matches
        #     inlier_matches = 0
        #     for m in good_matches:
        #         src_pt = np.float32([keypoints_small[m.queryIdx].pt]).reshape(-1, 1, 2)
        #         dst_pt = np.float32([keypoints_large[m.trainIdx].pt]).reshape(-1, 1, 2)
        #         transformed_pt = cv2.perspectiveTransform(src_pt, M)
        #         distance = cv2.norm(dst_pt, transformed_pt, cv2.NORM_L2)
        #         if distance < 5.0:
        #             inlier_matches += 1

        #     # Calculate the inlier ratio as a similarity score
        #     inlier_ratio = inlier_matches / len(good_matches)
        #     normalized_score = inlier_ratio / len(keypoints_large)
        print("score = ", max_similarity)
        # print("Normalized score = ", normalized_score)
        x.append(matching.shape[1])
        y.append(max_similarity)
        # result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX, -1)
        # heat = (result*255).astype(np.uint8)
        # heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        # heat = cv2.resize(heat, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        # template1 = np.stack((template,) * 3, axis=-1)
        # tmp = np.concatenate([template1, heat], axis= 0)
        # cv2.imshow("compare", tmp)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    print("scale and score = ", x , y)
    # Set the window size for the moving average
    window_size = 5
    # Calculate the moving average for x and y data
    x_ma = np.convolve(x, np.ones(window_size), 'valid') / window_size
    y_ma = np.convolve(y, np.ones(window_size), 'valid') / window_size
    plt.plot(x, y)
    plt.plot(x_ma, y_ma)
    plt.xlabel("scale length (px) ")
    plt.ylabel("Minimum distance from the template")
    # plt.show()
    plt.savefig("../../test_dis/feature_matching/new/match_10_size_90.png")
    # plt.plot(x,y_normal)
    # plt.savefig("./normalized_120.png")
    # plt.show()
    # Display the result
    # cv2.imshow("Feature Matching Result", matching_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()





if __name__ == "__main__":
    main()