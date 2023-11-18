// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

#include <common/shader.hpp>
#include <common/texture.hpp>
#include <common/controls.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <include/global.hpp>

GLFWwindow* window;

bool initializeGLFWAndGLEW(GLFWwindow** window, int width, int height);
bool initializeGLFW();
void setGLFWWindowHints();
bool createGLFWWindow(GLFWwindow** window, int width, int height);
bool initializeGLEW();
void setupOpenGLRendering();
void getWidthandHeight(cv::Mat image, int* width, int* height);
// render object 
void render(GLuint programID, GLuint MatrixID, glm::mat4 MVP, GLuint vertexbuffer, GLuint colorbuffer);
glm::mat4 getMVP();
glm::mat4 getMVPMatrix(const cv::Vec3d& rvec, const cv::Vec3d& tvec, 
                       const cv::Mat& cameraMatrix, int width, int height);


int main() {
    // Set coordinate system
    cv::Mat objPoints(4, 1, CV_32FC3);
    objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength/2.f, markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength/2.f, markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength/2.f, -markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength/2.f, -markerLength/2.f, 0);

	cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);

    cv::VideoCapture inputVideo;
    inputVideo.open(0);
	cv::Mat image, imageCopy;
	
	int width, height;
	inputVideo.grab();
	inputVideo.retrieve(image);

	getWidthandHeight(image, &width, &height);

	// initilize openGL
    if (!initializeGLFWAndGLEW(&window, width/2, height/2)) {
		return -1;
	}
	setupOpenGLRendering();

    GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

    // Create and compile our GLSL program from the shaders
	GLuint programID = LoadShaders( 
		"../shaders/TransformVertexShader.vertexshader", 
		"../shaders/ColorFragmentShader.fragmentshader" 
		);

	// Use our shader
	glUseProgram(programID);
	// Get a handle for our "MVP" uniform
	GLuint MatrixID = glGetUniformLocation(programID, "MVP");

    glm::mat4 MVP = getMVP();

    GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	GLuint colorbuffer;
	glGenBuffers(1, &colorbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_color_buffer_data), g_color_buffer_data, GL_STATIC_DRAW);


	GLuint fbo, renderedTexture;

	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	// Create a texture to render to
	glGenTextures(1, &renderedTexture);
	glBindTexture(GL_TEXTURE_2D, renderedTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	// Set texture as color attachment
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderedTexture, 0);

	GLenum drawBuffers[1] = {GL_COLOR_ATTACHMENT0};
	glDrawBuffers(1, drawBuffers); // "1" is the size of DrawBuffers

	// Check if FBO is complete
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		std::cerr << "Error: Framebuffer is not complete!" << std::endl;
		return -1; // Or handle the error appropriately
	}

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    while (inputVideo.grab()) {

        inputVideo.retrieve(image);
        image.copyTo(imageCopy);
        
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;

        detector.detectMarkers(image, corners, ids);
        // If at least one marker detected
        if (!ids.empty()) {

            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
            int nMarkers = corners.size();
            std::vector<cv::Vec3d> rvecs(nMarkers), tvecs(nMarkers);

            // Calculate pose for each marker
            for (int i = 0; i < nMarkers; i++) {
                cv::solvePnP(objPoints, corners.at(i), cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
                // std::cout << " " << tvecs[i][0] << " " << tvecs[i][1] << " " << tvecs[i][2] << std::endl;
				cv::drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 6);
				MVP = getMVPMatrix(rvecs[i], tvecs[i], cameraMatrix, width, height);

				render(programID, MatrixID, MVP, vertexbuffer, colorbuffer);

				// Read the pixels from the texture into an OpenCV Mat
				cv::Mat renderedImage(height, width, CV_8UC3);
				glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, renderedImage.data);
				cv::flip(renderedImage, renderedImage, 0); // Flip the image vertically

				cv::addWeighted(imageCopy, 0.8, renderedImage, 1.0, 0, imageCopy);
            }

        }
        // Show resulting image and close window
        cv::imshow("out", imageCopy);
        char key = (char) cv::waitKey(1);
        if (key == 27)
            break;
    }
}

bool initializeGLFWAndGLEW(GLFWwindow** window, int width, int height) {
    if (!initializeGLFW()) return false;
    setGLFWWindowHints();
    if (!createGLFWWindow(window, width, height)) return false;
    if (!initializeGLEW()) return false;
    return true;
}
bool initializeGLFW() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
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
	// glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
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

bool initializeGLEW() {
    glewExperimental = GL_TRUE;  // Enable GLEW experimental features
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return false;
    }
    return true;
}

void setupOpenGLRendering() {
    // Dark blue background
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
}

void render(GLuint programID, GLuint MatrixID, glm::mat4 MVP, GLuint vertexbuffer, GLuint colorbuffer) {
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

// getMVP return the mat4 MVP
glm::mat4 getMVP() {
    // Projection matrix : 45� Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	glm::mat4 Projection = glm::perspective(glm::radians(45.0f), 16.0f / 9.0f, 0.1f, 100.0f);
	// Camera matrix
	glm::mat4 View       = glm::lookAt(
								glm::vec3(0,0,-3), // Camera is at (4,3,-3), in World Space
								glm::vec3(0,0,0), // and looks at the origin
								glm::vec3(0,1,0)  // Head is up (set to 0,-1,0 to look upside-down)
						   );
	// Model matrix : an identity matrix (model will be at the origin)
	glm::mat4 Model      = glm::mat4(1.0f);
	// Our ModelViewProjection : multiplication of our 3 matrices
	glm::mat4 MVP        = Projection * View * Model; // Remember, matrix multiplication is the other way around
    return MVP;

}

glm::mat4 getMVPMatrix(const cv::Vec3d& rvec, const cv::Vec3d& tvec, const cv::Mat& cameraMatrix, int width, int height) {
    float nearPlane = 0.01f;
    float farPlane = 100.0f;

    // Convert rotation vector to rotation matrix
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // Scale matrix
    // Scale down the box to match the marker size
    glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(4.0f)); // Scale down by a factor of 0.25

    // Rotation matrix
    glm::mat4 rotation = glm::mat4(1.0f);
    rotation[0][0] = static_cast<float>(R.at<double>(0, 0));
    rotation[0][1] = static_cast<float>(-R.at<double>(1, 0)); // Invert the y-axis
    rotation[0][2] = static_cast<float>(-R.at<double>(2, 0)); // Invert the z-axis

    rotation[1][0] = static_cast<float>(R.at<double>(0, 1));
    rotation[1][1] = static_cast<float>(-R.at<double>(1, 1)); // Invert the y-axis
    rotation[1][2] = static_cast<float>(-R.at<double>(2, 1)); // Invert the z-axis

    rotation[2][0] = static_cast<float>(R.at<double>(0, 2));
    rotation[2][1] = static_cast<float>(-R.at<double>(1, 2)); // Invert the y-axis
    rotation[2][2] = static_cast<float>(-R.at<double>(2, 2)); // Invert the z-axis

    // Translation matrix
    glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(tvec[0], -tvec[1], -tvec[2]));
    // glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, -20));
    
    glm::mat4 model = translation * rotation * scale;

    // Projection matrix based on camera intrinsics
    float fx = static_cast<float>(cameraMatrix.at<double>(0, 0));
    float fy = static_cast<float>(cameraMatrix.at<double>(1, 1));
    float cx = static_cast<float>(cameraMatrix.at<double>(0, 2));
    float cy = static_cast<float>(cameraMatrix.at<double>(1, 2));

    glm::mat4 projection = glm::mat4(1.0f);
    projection[0][0] = 2.0f * fx / width;
    projection[1][1] = 2.0f * fy / height;
    projection[2][2] = -(farPlane + nearPlane) / (farPlane - nearPlane);
    projection[2][3] = -1.0f;
    projection[3][2] = -2.0f * farPlane * nearPlane / (farPlane - nearPlane);
    projection[0][2] = 1.0f - 2.0f * cx / width;
    projection[1][2] = 1.0f - 2.0f * cy / height;

    // glm::mat4 projection = glm::perspective(glm::radians(45.0f), 16.0f / 9.0f, 0.1f, 100.0f);

    glm::mat4 View       = glm::lookAt(
                            glm::vec3(0,0,0), // Camera is at (4,3,-3), in World Space
                            glm::vec3(0,0,-1), // and looks at the origin
                            glm::vec3(0,1,0)  // Head is up (set to 0,-1,0 to look upside-down)
                        );

    // Combine model and projection matrices
    glm::mat4 MVP = projection * View * model;

    return MVP;
}

void getWidthandHeight(cv::Mat image, int* width, int* height) {
    if (width != nullptr && height != nullptr) {
        *width = image.cols;
        *height = image.rows;
        std::cout << "width: " << *width << " height: " << *height << std::endl;
    }
}

