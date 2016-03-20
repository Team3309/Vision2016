#ifndef TARGET_H
#define TARGET_H

class Position {
  double x, y;
};

class Size {
  double width, height;
};

class Target {
  Position pos;
  Size size;
  double distance;
  double elevationAngle;
  double azimuth;
};

#endif
