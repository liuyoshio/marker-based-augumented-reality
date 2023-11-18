#ifndef OPENGL_UTILS_HPP
#define OPENGL_UTILS_HPP

#include <GL/glew.h>
#include <GLFW/glfw3.h>

bool initializeOpenGL(GLFWwindow** window, int width, int height);
void setupOpenGLRendering();
void renderScene();

#endif // OPENGL_UTILS_HPP
