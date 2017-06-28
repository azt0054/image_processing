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
from features import shapedetector as sd
from features import sobeledge as se
from features import maskverifier as mv



class CVFilter:
    def __init__(self,org_filename):
        self.org_filename = org_filename
        self.org_image = cv2.imread(org_filename)
        self.processed = False
        self.process_info = []
        if self.org_image is not None:
            self.image_grayscale = cv2.imread(org_filename,0)
            self.is_default = isDefaultImage(self.image_grayscale)
            self.height,self.width = self.org_image.shape[:2]
            self.background = False
            self.sd = sd.ShapeDetector()
            if postpone_larger_files and (self.height>800 or self.width>800):
                raise ValueError("too big!",self.org_filename)
        else:
            raise ValueError("not a valid image file!",self.org_filename)


    def save(self):
        if self.processed:
            mod_filename = os.path.join(test_image_dir,self.org_filename.split("/")[-1])#processed_image_dir
            final_image = resize(self.image_list[-2])
            """image_stages = vStack(hStack(*self.image_list[:3]), \
                    hStack(*self.image_list[3:6]),)"""
            cv2.imwrite(mod_filename,final_image)
            self.comparisionImage()
        else:
            log("processing failed!")


    def comparisionImage(self):
        if self.processed:
            comp_filename = os.path.join(comparision_image_dir,self.org_filename.split("/")[-1])
            final_image = resize(self.image_list[-2])
            comp_image =  hStack(resize(self.org_image,0),final_image)
            cv2.imwrite(comp_filename,comp_image)


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


    def moveBackgroundImages(self,background_less = 0):
        if self.background:
            mod_filename = os.path.join(background_image_dir,self.org_filename.split("/")[-1])
            copyfile(self.org_filename, mod_filename)
        elif background_less:
            mod_filename = os.path.join(backgroundless_image_dir,self.org_filename.split("/")[-1])
            copyfile(self.org_filename, mod_filename)


    def hasBackground(self):
        min_hsv, max_hsv, edge_hsv_info,color_info  = getBorderInfo(self.org_image)
        #print min_hsv,max_hsv,edge_hsv_info
        if edge_depth:
            self.background = self.checkCorners(min_hsv,max_hsv,color_info) or  self.checkEdges(edge_hsv_info,color_info)
        else:
            self.background =  self.checkCorners(min_hsv,max_hsv,color_info)
        return self.background


    def checkCorners(self,min_hsv,max_hsv,color_info):
        hsv_status = self.checkCornersHSV(min_hsv,max_hsv)
        color = ''
        for pos in ['TL','TR','BR','BL']:
            roi = color_info[pos]
            if max(roi['min'])<250 and min(roi['min'])>200:
                if not color or (len(roi['color'])>len(color) and color in roi['color']):
                    color = roi['color']
                elif roi['color'] not in color:
                    #log("invalid color"):
                    color = ""
                    break
            else:
                color = ""
                break
        return hsv_status and (color and color!='white')


    def checkEdges(self,edge_hsv_info,color_info):
        hsv_status = self.checkEdgesHSV(edge_hsv_info)
        color = ''
        for pos in ['L','T','R','B']:
            roi = color_info[pos]
            if max(roi['min'])<250 and min(roi['min'])>200:
                if not color or (len(roi['color'])>len(color) and color in roi['color']):
                    color = roi['color']
                elif roi['color'] not in color:
                    #log("invalid color"):
                    color = ""
                    break
            else:
                color = ""
                break
        return hsv_status and (color and color!='white')


    def checkCornersHSV(self,min_hsv,max_hsv):
        if self.org_filename.endswith('png'):
            if min_hsv==max_hsv==[0, 0, 0] or min_hsv==max_hsv==[0, 0, 255]:                                                           # have white bg on all corners
                return False
            elif min_hsv==[0, 0, 0] or max_hsv==[0, 0, 255]:
                return False
            elif min_hsv==max_hsv:
                return False
            else:
                return True
        else:
            if min_hsv[:2]==max_hsv[:2]==[0, 0,] and min_hsv[2]>250 and max_hsv[2] > 250:                                                           # have white bg on all corners
                #log("has white background on the edges/ cropped already")
                return False
            elif (min_hsv[0]==min_hsv[1]==0 and max_hsv[2]>250 and min_hsv[2]>240 and max_hsv[0]>30):   # have very solid color close to edge or 100% cropped
                #log("has very solid color close to the edge/ cropped already")
                return False
            elif min(max_hsv)>127:
                return False
            elif sorted(min_hsv)[1]>127:
                return False
            elif (max_hsv[1]>127 and min_hsv[1]<4):
                #log("has solid color close to the edge/ cropped already")
                return False
            elif min_hsv==max_hsv:
                if min_hsv[0] > 80 and min_hsv[2]==255:
                    return False
                elif max_hsv[1]>80 and min_hsv[2] > 250:
                    return False
            return True

    def checkEdgesHSV(self,edge_hsv_info):
        left,top,right,bottom = edge_hsv_info
        background_color_tuple = ([0, 0, 255], [0, 0, 255])
        shape, corners = self.getShape()
        if left==right==background_color_tuple:
            if top!=background_color_tuple and bottom!=background_color_tuple:
                if shape in ('rectangle',) and len(corners)<5:
                    if min(top[1])<127 and min(bottom[1])<127:
                        return True
            return False
        elif top==bottom==background_color_tuple:
            if left!=background_color_tuple and right!=background_color_tuple:
                if shape in ('rectangle',) and len(corners)<5:
                    if min(left[1])<127 and min(right[1])<127:
                        return True
            return False
        else:
            return False

    def getShape(self):
        _,thresholded = thresh(self.image_grayscale,240)
        contours = getContours(thresholded,2)
        contours = getSortedContourInfo(contours)
        shape = self.sd.detect(contours[0])
        image = getImageFromContours(self.image_grayscale,[contours[0]])
        #_,image = getSmallestConvexMask(self.image_grayscale,contours=[contours[0]])
        dst = cv2.cornerHarris(image,2,3,0.04)
        dst = cv2.dilate(dst,None)
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(image,np.float32(centroids),(5,5),(-1,-1),criteria)
        return shape,corners


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

    def reverseProcessGamma(self,mask_info):
        gamma,thresholded_gamma = gc.getGamma(self.org_image, gamma_limit)
        masked_gamma = cv2.subtract(thresholded_gamma,mask_info["full_mask"])

        inverse_thresholded_gamma = erode(thresh(thresholded_gamma)[1])

        flood_filled_inverse_mask = customFill(morphOpen(inverse_thresholded_gamma))
        flood_filled_inverse_mask = customFill(flood_filled_inverse_mask,(0,1))

        gamma_contours = getContours(flood_filled_inverse_mask,1)
        gamma_contours = simpleContourFilter(gamma_contours, min_contour_size)
        if not gamma_contours:
            log("no significant contour found")
            return

        #flood_filled_inverse_mask_3D = dStack(flood_filled_inverse_mask)[0]
        flood_filled_inverse_gamma = cv2.subtract(masked_gamma,flood_filled_inverse_mask)

        #flood_filled_inverse_gamma_2D = get2DFrom3D(flood_filled_inverse_gamma)
        inverse_contours = getContours(flood_filled_inverse_gamma,1)
        inverse_contours = getCenterContour(self.org_image,inverse_contours)[:1]

        if len(inverse_contours):
            grouped_contour = groupContours(inverse_contours,distance=0)
            top_3 = np.vstack(list(grouped_contour[:3]))
            #xq,yq,wq,hq = cv2.boundingRect(top_3)
            #approx_contour_outline = getContourApprox(inverse_contours[0])
            grouped_contour_hull = cv2.convexHull(top_3)
            reverse_gamma = getImageFromContours(self.org_image,inverse_contours)
            #blank_image = np.zeros((self.height,self.width,3), np.uint8)

            contours_mask = getContours(flood_filled_inverse_mask,2)
            for cnt in groupContours(contours_mask,distance=0):
                status = checkIfInside(grouped_contour_hull,cnt)
                if status[0]:
                    cv2.drawContours(reverse_gamma,[cnt],-1,(255,255,255),-1)
            return reverse_gamma


    def rectGrabcut(self,foreground_rect,mask_info=None):
        grabcut_image = np.zeros([self.height,self.width],np.uint8)
        mask = np.zeros([self.height,self.width],dtype = np.uint8)
        if mask_info:
            if mask_info.get("grabcut_rectangle",False):
                foreground_rect = mask_info["grabcut_rectangle"]
                wd,ht = foreground_rect[2:4]
                #print foreground_rect,self.width,self.height
                if not tight_mask and (wd*ht)/float(self.height*self.width) >0.97:
                    self.process_info.append("novalidmask")
                    log("novalidmask!")
                    return self.org_image,self.org_image
                elif not tight_mask and self.height>1000 and self.width>1000 and (wd*ht)/float(self.height*self.width)>0.85:
                    self.process_info.append("novalidmask")
                    log("novalidmask!")
                    return self.org_image,self.org_image

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
        self.computed_foreground = np.where((mask==1) + (mask==3),255,0).astype('uint8')
        self.computed_background = np.where(dilate(self.computed_foreground,2)==255,0,255).astype('uint8')

        grabcut_image = cv2.bitwise_and(self.org_image,self.org_image,mask=dilate(self.computed_foreground,3))
        self.computed_background = cv2.cvtColor(self.computed_background, cv2.COLOR_GRAY2BGR)

        grabcut_normal = cv2.add(grabcut_image,self.computed_background)
        grabcut_blended = self.blendMask(grabcut_image,self.computed_background)
        return grabcut_normal,grabcut_blended


    def fixBackground(self):
        try:
            gamma,thresholded_gamma = gc.getGamma(self.org_image, gamma_limit)
            thresholded_gamma_edges,thresholded_gamma_auto_edges = ce.getCannyEdge(thresholded_gamma)

            mask,mask_info = self.getCompleteMask()
            _,inverse_mask = thresh(mask)

            x,y,width,height,rectangle = getSmallestRect(inverse_mask,1)

            """reverse_gamma = self.reverseProcessGamma(mask_info)
            if reverse_gamma.any():
                x_rg,y_rg,width_rg,height_rg,rectangle_rg = getSmallestRect(reverse_gamma,1)
                #show(hStack(self.org_image,mask_info['sure_foreground'],reverse_gamma),5000)"""

            normal_grabcut,blended_grabcut  = self.rectGrabcut((x,y,width,height),mask_info)

            #mv.checkValidMask(self.computed_foreground)

            self.processed = True

            mod_filename = os.path.join(test_image_dir,self.org_filename.split("/")[-1])
            soble_edge,laplacian = se.sobelEdge(self.org_filename)
            """if reverse_gamma.any() and self.processed:
                final_image = vStack(hStack(self.org_image,blended_grabcut,mask_info['sure_foreground']), \
                                hStack(reverse_gamma,soble_edge,laplacian))
            else:
                final_image = vStack(hStack(self.org_image,mask_info['sure_foreground']), \
                                hStack(soble_edge,laplacian))
            cv2.imwrite(mod_filename,final_image)"""

            self.image_list = [self.image_grayscale,self.complete_contour_image,inverse_mask, \
                            normal_grabcut,blended_grabcut,self.org_image]
            self.image_names = ["GrayScale","GammaCorrect","Mask", \
                            "Grabcut","Processed","Original"]
        except ValueError,e:
            if e.message.startswith("timeout"):
                log("taking too long!",self.org_filename)
            else:
                log("bad cutting sample!",self.org_filename)
        except:
            #traceback.print_exc()
            log("bad cutting sample!",self.org_filename)


if __name__ == '__main__':
    org_filename = 'images/original/12545.jpg'
    cv_filter = CVFilter(org_filename)
    if cv_filter.hasBackground():
        cv_filter.fixBackground()
        cv_filter.display()
