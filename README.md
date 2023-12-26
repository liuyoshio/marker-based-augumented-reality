# marker-based-augumented-reality
## Implementation of a marker based augumenty reality project based on cpp, OpenCv and OpenGL.

The camera is pre-calibrated to get the intrincs. You need to calibrate for your own camera.
For each frame, the marker is detected and the RT matrix is calculated by SolvePnP. Then display the cube on the detected marker according to its estimated position.

**Marker used**: ArUco

## How to run
For building, use Cmake


![image](./demo/demo1.jpg)
