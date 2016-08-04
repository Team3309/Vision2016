# Vision2016
FRC Team 3309's vision tracking code for the 2016 game, Stronghold.

Overview
========

- Written Python
- Works with Raspberry Pi Camera Module or more generic `cv2.VideoCapture`
- Built with OpenCV
- Web interface for calibrating thresholds and camera gains (if using raspberry pi camera)

How it works
============

1. Camera image retrieved using `cv2.VideoCapture` or `PiCamera`
2. Image converted from BGR to HSV
3. Image converted to binary with the given HSV thresholds
4. Erode + Dilate image to clean up some artifacts / erroneous blobs
5. Find contours with OpenCV
6. Score contours and remove ones below a certain score
7. Calculate target's physical pose (distance, azimuth, and elevation angle)
7. Send results to roboRIO and web interface

Scoring a contour
-----------------

Each contour/blob detected by OpenCV is assigned a score according to how much it matches features of a Stronghold goal.

1. Blobs must meet a certain minimum area in pixels
2. Aspect ratio must be close to 20/14 (width / height) of goal marker
3. Coverage score must be close to 1/3 (coverage is the amount of filled pixels compared to the bounding box)
4. Check `cv2.HuMoments`, `hu[6]` should be close to 0
5. "Profile" the marker by comparing how well the shape matches the actual marker (two vertical lines on the outside connected by one line across the bottom)

Networking
----------

UDP packets containing JSON describing the found targets are sent to the roboRIO as each image is processed.  

The web interface uses websockets to communicate JSON target information as well as JPEG versions of the camera image at various stages in the processing pipeline.
Websockets are also used to adjust the running configuration, which is saved to a JSON file when changes are made to persist between runs.
