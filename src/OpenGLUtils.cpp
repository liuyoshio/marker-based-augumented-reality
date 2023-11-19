#include "OpenGLUtils.hpp"

bool initOpenGL(GLFWwindow** window, int width, int height) {
    if (!initGLFW()) return false;
    setGLFWWindowHints();
    if (!createGLFWWindow(window, width, height)) return false;
    if (!initGLEW()) return false;
    return true;
}

bool initGLFW() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    return true;
}

bool initGLEW() {
    glewExperimental = GL_TRUE;  // Enable GLEW experimental features
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return false;
    }
    return true;
}

void setGLFWWindowHints() {
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	// make glfw window invisiale
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
}

bool createGLFWWindow(GLFWwindow** window, int width, int height) {
    *window = glfwCreateWindow(width, height, "OpenGL Window", nullptr, nullptr);
    if (!*window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(*window);
    return true;
}

void setOpenGLRendering(GLFWwindow* window) {
    // Dark blue background
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
}

bool setFrameBuffer(GLuint &fbo, GLuint &renderedTexture, int width, int height) {
    // Generate and bind the Framebuffer
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // Generate and bind the texture
    glGenTextures(1, &renderedTexture);
    glBindTexture(GL_TEXTURE_2D, renderedTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // Attach the texture to the framebuffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderedTexture, 0);
    GLenum drawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, drawBuffers);

    // Check if the framebuffer is complete
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Error: Framebuffer is not complete!" << std::endl;
        return false; // Return false to indicate failure
    }
    return true; // Return true to indicate success
}

void renderScene(GLFWwindow* window, GLuint programID, GLuint MatrixID, glm::mat4 MVP, GLuint vertexbuffer, GLuint colorbuffer) {
    		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Send our transformation to the currently bound shader, 
		// in the "MVP" uniform
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

		// 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

		// 2nd attribute buffer : colors
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

		// Draw the triangle !
		glDrawArrays(GL_TRIANGLES, 0, 12*3); // 12*3 indices starting at 0 -> 12 triangles

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

}