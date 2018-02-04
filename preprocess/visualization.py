#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/17/18 10:36 AM
# @Author  : gehen
# @File    : visualization.py
# @Software: PyCharm Community Edition

import cv2
import colorsys

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)

def draw_tracker(image,tracks):
    if len(tracks)>0:
        for track in tracks:
            if not track.is_confirmed():
                continue
            person_id = track.track_id
            id_color = create_unique_color_uchar(person_id)
            bbox = map(int,track.to_tlwh())
            image = cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2]+bbox[0],bbox[3]+bbox[1]),id_color,2)
            image = cv2.putText(image,'{}'.format(person_id),(bbox[0],bbox[1]-10),cv2.FONT_ITALIC,0.8,[0,0,0],2)
    return image
