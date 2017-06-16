import os
import cv2
import numpy as np
import pdb
import traceback
from shutil import *

from config import *
from helper import *
from contour_tools import *
from mask_tools import *
from features import gammacorrection as gc
from features import cannyedge as ce


class CVFilter:
    def __init__(self,org_filename):
        self.org_filename = org_filename
        self.org_image = cv2.imread(org_filename)
        self.processed = False
        if self.org_image is not None:
            self.image_grayscale = cv2.imread(org_filename,0)
            self.height,self.width = self.org_image.shape[:2]
        else:
            raise ValueError("not a valid image file!")


    def save(self):
        if self.processed:
            mod_filename = os.path.join(processed_image_dir,self.org_filename.split("/")[-1])
            total_images = len(self.image_list)
            if total_images>2:
                image_stages = vStack(hStack(*self.image_list[:total_images/2]), \
                        hStack(*self.image_list[total_images/2:total_images]),)
            else:
                image_stages = hStack(*self.image_list)
            cv2.imwrite(mod_filename,image_stages)
        else:
            log("processing failed!")


    def display(self):
        if self.processed:
            for im in range(len(self.image_list)):
                addText(self.image_list[im],self.image_names[im])
            image_stages = vStack(hStack(*self.image_list[:3]), \
                    hStack(*self.image_list[3:6]),)
            cv2.imshow("CVFilter", cv2.resize(image_stages,(900,600)))
            k = cv2.waitKey(wait_time)
            if k==27:
                cv2.destroyAllWindows()
            elif k==ord('q'):
                sys.exit()
        else:
            log("processing failed!")


    def moveBackgroundImages(self):
        if self.hasBackground():
            mod_filename = os.path.join(background_image_dir,self.org_filename.split("/")[-1])
            copyfile(self.org_filename, mod_filename)
        else:
            mod_filename = os.path.join(backgroundless_image_dir,self.org_filename.split("/")[-1])
            copyfile(self.org_filename, mod_filename)


    def hasBackground(self):
        min_hsv, max_hsv = getHSV(self.org_image)
        if min_hsv==max_hsv==[0, 0, 255]:                                                           # have white bg on all corners
            #log("has white background on the edges/ cropped already")
            return False
        elif max_hsv[1]>80 or (min_hsv[0]==min_hsv[1]==0 and max_hsv[2]==255 and min_hsv[2]>220):   # have very solid color close to edge or 100% cropped
            #log("has very solid color close to the edge/ cropped already")
            return False
        elif max_hsv[1]>40 and min_hsv[1]==0:                                                       # have solid color close to the edge and may also be cropped
            #log("has solid color close to the edge/ cropped already")
            return False
        else:
            return True


    def blendMask(self,image,mask):
        _,image_inverse = thresh(mask)
        image_edge = cv2.subtract(dilate(image_inverse,2), image_inverse)
        image_edge_blurred = cv2.GaussianBlur(image_edge, (15, 15), 0)

        mask_edge = cv2.subtract(mask, erode(mask,2))
        mask_edge_blurred = cv2.GaussianBlur(mask_edge, (15, 15), 0)

        connect_image_mask_edge = cv2.add(cv2.subtract(dilate(image_inverse,2), dilate(image_inverse,1)), \
                                    cv2.subtract(mask, erode(mask,1)))

        image_connector = cv2.addWeighted(image_edge_blurred,0.8,connect_image_mask_edge,0.5,0)
        mask_connector = cv2.addWeighted(mask_edge_blurred,0.8,connect_image_mask_edge,0.5,0)
        image_mask_connector = cv2.addWeighted(image_connector,0.8,mask_connector,0.6,0)

        blended_image = cv2.addWeighted(image,1,image_mask_connector,0.8,0)
        blended_mask = cv2.addWeighted(mask,1,image_mask_connector,0.8,0)

        blended_image_mask = cv2.add(blended_image,blended_mask)
        return blended_image_mask


    def addMissingEdges(self,current_contour,previous_contour=None):
        current_contour_image = getImageFromContours(self.org_image,current_contour)
        complete_contours = list(getContours(current_contour_image,contour_type=2))
        if previous_contour:
            previous_contour_image = getImageFromContours(self.org_image,previous_contour)
            extra_edges = cv2.subtract(previous_contour_image,current_contour_image)
            extra_edges_contours = getContours(extra_edges)
            for extra_edge in extra_edges_contours:
                for cnt in complete_contours:
                    status,distance_info = checkIfInside(cnt,extra_edge,min_inside=0,distance=True)
                    sorted_distance_info = sorted(distance_info, reverse=True)
                    if status:
                        complete_contours.append(extra_edge)
                        break
                    elif sorted_distance_info[0]>=-1.5 and cv2.contourArea(extra_edge)>max_contour_size:
                        if abs(sorted_distance_info[-1])<max_contour_distance:
                            complete_contours.append(extra_edge)
                            break

            complete_contours_image = getImageFromContours(self.org_image,complete_contours)
        return complete_contours


    def getEdgeContours(self,edge_threshold=min_edge_threshold_limit):
        bottom_mask = self.getBasicMask()
        edge,auto = ce.getCannyEdge(self.org_image,edge_threshold)

        bottom_masked_edge = cv2.subtract(edge,get2DFrom3D(bottom_mask))
        dilated_edge = dilate(bottom_masked_edge, 5)

        edge_contours = getContourFromImage(dilated_edge)
        filtered_contours = simpleContourFilter(edge_contours,max_contour_size)

        distance_filtered_contours = distanceFilter(filtered_contours,max_contour_distance)
        return distance_filtered_contours


    def getBasicMask(self):
        self.basic_edge,auto = ce.getCannyEdge(self.org_image,max_edge_threshold_limit)
        dilated_edge = dilate(self.basic_edge, 1)

        edge_contours = getContourFromImage(dilated_edge)
        filtered_contours = simpleContourFilter(edge_contours,min_contour_size)
        self.basic_contour_image = getImageFromContours(self.org_image,filtered_contours)

        x,y,wd,ht,new_image = getSmallestRect(self.basic_contour_image,0)
        mask = createRectCuttingMask(new_image,x,y,wd,ht)
        blank_image = np.zeros((self.height,self.width,3), np.uint8)
        new_mask = createRectCuttingMask(blank_image,0,y+ht,self.width,(self.height-y-ht))
        bottom_mask = cv2.subtract(mask,new_mask)
        return bottom_mask


    def getCompleteMask(self):
        mask_info = {}
        filtered_contours = self.getEdgeContours(min_edge_threshold_limit)
        filtered_contours_prev = self.getEdgeContours(min_edge_threshold_limit-10)
        full_contours = self.addMissingEdges(filtered_contours,filtered_contours_prev)

        self.complete_contour_image = getImageFromContours(self.org_image,full_contours)
        x,y,wd,ht,new_image = getSmallestRect(erode(self.complete_contour_image,2),0)
        full_mask = createRectCuttingMask(new_image,x,y,wd,ht)

        _,full_mask = getSmallestConvexMask(self.complete_contour_image,max_contour_size,contours=full_contours)

        mask_info["grabcut_rectangle"] = (x,y,wd,ht)
        mask_info["full_mask"] = full_mask
        mask_info["sure_foreground"] = erode(self.complete_contour_image,6)
        mask_info["sure_background"] = erode(full_mask,7)
        probable_foreground = cv2.subtract(self.complete_contour_image,erode(self.complete_contour_image,5))
        mask_info["probable_foreground"] = probable_foreground
        return full_mask,mask_info


    def rectGrabcut(self,foreground_rect,mask_info=None):
        grabcut_image = np.zeros([self.height,self.width],np.uint8)
        mask = np.zeros([self.height,self.width],dtype = np.uint8)
        if mask_info:
            if mask_info.get("grabcut_rectangle",False):
                foreground_rect = mask_info["grabcut_rectangle"]

        masked = 0
        for cut in range(grabcut_limit):
            mask = applyMaskRules(mask,mask_info,probables=True)
            bgdmodel = np.zeros((1,65),np.float64)
            fgdmodel = np.zeros((1,65),np.float64)
            if masked:
                cv2.grabCut(self.org_image,mask,foreground_rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
            else:
                cv2.grabCut(self.org_image,mask,foreground_rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
                masked+=1
            if cut<grabcut_limit-1:
                mask = applyMaskRules(mask,mask_info)
        computed_foreground = np.where((mask==1) + (mask==3),255,0).astype('uint8')
        computed_background = np.where(dilate(computed_foreground,2)==255,0,255).astype('uint8')

        grabcut_image = cv2.bitwise_and(self.org_image,self.org_image,mask=dilate(computed_foreground,3))
        computed_background = cv2.cvtColor(computed_background, cv2.COLOR_GRAY2BGR)

        grabcut_normal = cv2.add(grabcut_image,computed_background)
        grabcut_blended = self.blendMask(grabcut_image,computed_background)
        return grabcut_normal,grabcut_blended


    def fixBackground(self):
        try:
            gamma,thresholded_gamma = gc.getGamma(self.org_image, gamma_limit)
            thresholded_gamma_edges,thresholded_gamma_auto_edges = ce.getCannyEdge(thresholded_gamma)

            mask,mask_info = self.getCompleteMask()
            _,inverse_mask = thresh(mask)

            x,y,width,height,rectangle = getSmallestRect(inverse_mask,1)
            normal_grabcut,blended_grabcut  = self.rectGrabcut((x,y,width,height),mask_info)
            self.processed = True

            self.image_list = [resize(self.org_image,0),resize(blended_grabcut)]
            self.image_names = ["Original","Processed"]
        except:
            print traceback.format_exc()


if __name__ == '__main__':
    org_filename = 'images/original/12545.jpg'
    cv_filter = CVFilter(org_filename)
    if cv_filter.hasBackground():
        cv_filter.fixBackground()
        cv_filter.display()
