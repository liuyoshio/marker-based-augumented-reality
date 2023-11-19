#ifndef OPENGL_UTILS_HPP
#define OPENGL_UTILS_HPP

#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

bool initOpenGL(GLFWwindow** window, int width, int height);
bool initGLFW();
bool initGLEW();
void setGLFWWindowHints();
bool createGLFWWindow(GLFWwindow** window, int width, int height);
void setOpenGLRendering(GLFWwindow* window);
bool setFrameBuffer(GLuint &fbo, GLuint &renderedTexture, int width, int height);
void renderScene(GLFWwindow* window, GLuint programID, GLuint MatrixID, glm::mat4 MVP, GLuint vertexbuffer, GLuint colorbuffer);

#endif
