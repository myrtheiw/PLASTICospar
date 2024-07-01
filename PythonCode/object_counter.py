# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:14:31 2024

@author: Henry Gunawan, Dimitris Mantas, Sam Parijs
"""

from collections import defaultdict

import cv2
import pandas as pd

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the Counter with default values for various tracking and counting parameters."""


        # Region & Line Information
        self.reg_pts = [(1400, 0), (1400, 1080)]
        self.line_dist_thresh = 15
        self.counting_region = None
        self.region_color = (255, 0, 255)
        self.region_thickness = 5

        # Image and annotation Information
        self.im0 = None
        self.tf = None
        self.view_img = False
        self.view_total_counts = True

        self.names = None  # Classes names
        self.annotator = None  # Annotator

        # Object counting Information
        self.count_txt_thickness = 0
        self.count_txt_color = (0, 0, 0)
        self.count_color = (255, 255, 255)

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False
        self.track_color = (0, 255, 0)

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        classes_names,
        reg_pts,
        count_reg_color=(255, 0, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_total_counts=True,
        draw_tracks=False,
        count_txt_thickness=1,
        count_txt_color=(0, 0, 0),
        count_color=(255, 255, 255),
        track_color=(0, 255, 0),
        region_thickness=2,
        line_dist_thresh=15,
    ):
        """
        Configures the Counter's image, bounding box line thickness, and counting region points.

        Args:
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): Flag to control whether to display the video stream.
            view_total_counts (bool): Flag to control whether to display the total counts on video stream.
            reg_pts (list): Initial list of points defining the counting region.
            classes_names (dict): Classes names
            track_thickness (int): Track thickness
            draw_tracks (Bool): draw tracks
            count_txt_thickness (int): Text thickness for object counting display
            count_txt_color (RGB color): count text color value
            count_color (RGB color): count text background color value
            count_reg_color (RGB color): Color of object counting region
            track_color (RGB color): color for tracks
            region_thickness (int): Object counting Region thickness
            line_dist_thresh (int): Euclidean Distance threshold for line counter
        """
        self.tf = line_thickness
        self.view_img = view_img
        self.view_total_counts = view_total_counts
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks
        

        # Object counting information
        self.total_counts  = 0
        self.count_class   = [0] * len(classes_names)
        self.conf_class    = [0] * len(classes_names)
        self.counting_list = []

        # Region and line selection
        if len(reg_pts) == 2:
            print("Line Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = LineString(self.reg_pts)
        else:
            print("Invalid Region points provided, region_points can be 2")
            print("Using Line Counter Now")
            self.counting_region = LineString(self.reg_pts)

        self.names = classes_names
        self.track_color = track_color
        self.count_txt_thickness = count_txt_thickness
        self.count_txt_color = count_txt_color
        self.count_color = count_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.line_dist_thresh = line_dist_thresh


    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""
        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        track_ids = tracks[0].boxes.id.int().cpu().tolist()
        confs = tracks[0].boxes.conf.cpu().tolist()

        # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)
        self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)

        # Extract tracks
        for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
            # Draw bounding box
            self.annotator.box_label(box, label=f"{track_id}:{self.names[cls]}", color=colors(int(cls), True))

            # Draw Tracks
            track_line = self.track_history[track_id]
            track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
            if len(track_line) > 30:
                track_line.pop(0)

            # Draw track trails
            if self.draw_tracks:
                self.annotator.draw_centroid_and_tracks(
                    track_line, color=self.track_color, track_thickness=self.track_thickness
                )

            prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

            # Count objects
            if len(self.reg_pts) == 2:
                if prev_position is not None:
                    distance = Point(track_line[-1]).distance(self.counting_region)
                    if distance < self.line_dist_thresh and track_id not in self.counting_list:
                        
                        self.counting_list.append(track_id)  # record all id
                        self.total_counts += 1               # count total object

                        #RECORD BASED ON CLASS
                        if 0 <= int(cls) < len(self.names):  # Check if cls is within the range of classes
                            self.count_class[int(cls)] += 1  # count object per class
                            self.conf_class[int(cls)] = ((self.count_class[int(cls)]-1) + conf)/self.count_class[int(cls)]  #count the average confidence per class
 


        #Print counting result on frames
        outcount_label = f"| Total Count : {self.total_counts} | "
        for i in range(len(self.names)):
            outcount_label += f"{self.names[i]} : {self.count_class[i]} | "

        # self.annotator.display_analytics(self.im0, outcount_label, self.count_txt_color, self.count_color, self.tf)

        

        # self.annotator.display_counts(
        #     counts=outcount_label,
        #     #count_txt_size=self.count_txt_thickness,
        #     count_txt_color=self.count_txt_color,
        #     count_bg_color=self.count_color,
        # )


    def export_to_csv(self):
        """Export count data to a CSV file."""
        # Export data to csv
        data = {'Class': range(len(self.count_class)), 'Class Name': list(self.names.values()), 'Count': self.count_class, 'Confidence': self.conf_class}
        count_class_df = pd.DataFrame(data)
        count_class_df.to_csv('count_class.csv', index=False)
        print("Data exported to CSV.")

    
    def display_frames(self):
        """Display frame."""
        if self.env_check:
            cv2.namedWindow("Ultralytics YOLOv8 Object Counter")
            cv2.imshow("Ultralytics YOLOv8 Object Counter", self.im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image

        if tracks[0].boxes.id is None:
            if self.view_img:
                self.display_frames()
            return
        self.extract_and_process_tracks(tracks)

        if self.view_img:
            self.display_frames()
        return self.im0


if __name__ == "__main__":
    ObjectCounter()
