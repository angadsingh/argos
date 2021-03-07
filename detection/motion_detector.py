import cv2
import imutils
import numpy as np

from detection.StateDetectorBase import StateDetectorBase


class SimpleMotionDetector(StateDetectorBase):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bg = None
        self.total_frames = 0

    def update_bg(self, gray):
        if self.config.md_reset_bg_model:
            self.bg = None
            self.total_frames = 0
            self.config.md_reset_bg_model = False

        # if the background model is None, initialize it
        if self.bg is None:
            self.bg = gray.copy().astype("float")
            return
        # update the background model by accumulating the weighted
        # average
        if self.config.md_update_bg_model or self.total_frames <= self.config.md_warmup_frame_count:
            cv2.accumulateWeighted(gray, self.bg, self.config.md_bg_accum_weight)

    def apply_mask_to_crop(self, minX, minY, maxX, maxY):
        is_contained = False

        mask_minx, mask_miny, mask_maxx, mask_maxy = self.config.md_mask
        if minX > mask_maxx or minY > mask_maxy or mask_minx > maxX or mask_miny > maxY:
            return None, is_contained
        else:
            # keep the part of crop, which is overlapped by mask
            minX = max(mask_minx, minX)
            maxX = min(mask_maxx, maxX)
            minY = max(mask_miny, minY)
            maxY = min(mask_maxy, maxY)

            if minX > mask_minx and minY > mask_miny and maxX < mask_maxx and maxY < mask_maxy:
                is_contained = True
        return (minX, minY, maxX, maxY), is_contained

    def show_masks(self, frame):
        if self.config.md_mask:
            if self.config.md_show_masks:
                minX, minY, maxX, maxY = self.config.md_mask
                cv2.rectangle(frame, (minX, minY), (maxX, maxY), (255, 255, 255), 1)

        if self.config.md_nmask:
            if self.config.md_show_masks:
                nminX, nminY, nmaxX, nmaxY = self.config.md_nmask
                cv2.rectangle(frame, (nminX, nminY), (nmaxX, nmaxY), (0, 0, 0), 1)

    def detect(self, frame):
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)

        self.total_frames += 1
        # whether part or whole of the motion occured outside the mask
        motion_outside = None

        # narrow the frame to a box with motion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame (disable by setting config.md_warmup_frame_count to -1
        if self.bg is not None and (self.total_frames > self.config.md_warmup_frame_count if self.config.md_warmup_frame_count > 0 else True):
            # compute the absolute difference between the background model
            # and the image passed in, then threshold the delta image
            delta = cv2.absdiff(self.bg.astype("uint8"), gray)
            thresh = cv2.threshold(delta, self.config.md_tval, 255, cv2.THRESH_BINARY)[1]
            # perform a series of erosions and dilations to remove small blobs
            if self.config.md_enable_erode:
                thresh = cv2.erode(thresh, None, iterations=self.config.md_erode_iterations)
            if self.config.md_enable_dilate:
                thresh = cv2.dilate(thresh, None, iterations=self.config.md_dilate_iterations)

            # find contours in the thresholded image and initialize the
            # minimum and maximum bounding box regions for motion
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if len(cnts) > 0:
                filter_pass = False
                for c in cnts:
                    if cv2.contourArea(c) > self.config.md_min_cont_area:
                        filter_pass = True
                        (x, y, w, h) = cv2.boundingRect(c)
                        (minX, minY) = (min(minX, x), min(minY, y))
                        (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))
                        if not self.config.md_blur_output_frame:
                            if self.config.md_show_all_contours:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                if filter_pass:
                    if self.config.md_blur_output_frame:
                        frame[minY:maxY, minX:maxX] = cv2.blur(frame[minY:maxY, minX:maxX], (83, 83))
                        if self.config.md_show_all_contours:
                            for c in cnts:
                                (x, y, w, h) = cv2.boundingRect(c)
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    crop = minX, minY, maxX, maxY
                    if self.config.md_mask:
                        crop, is_contained = self.apply_mask_to_crop(minX, minY, maxX, maxY)
                        if crop:
                            minX, minY, maxX, maxY = crop
                            motion_outside = not is_contained
                    if crop:
                        if maxX - minX > self.config.md_box_threshold_x and \
                                maxY - minY > self.config.md_box_threshold_y:
                            pass_nmask = True
                            if self.config.md_nmask:
                                nminX, nminY, nmaxX, nmaxY = self.config.md_nmask
                                if minX > nminX and minY > nminY and maxX < nmaxX and maxY < nmaxY:
                                    pass_nmask = False
                            if pass_nmask:
                                cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 2)
                                self.update_bg(gray)
                                return (frame, crop, motion_outside)

        self.update_bg(gray)
        return (frame, None, motion_outside)
