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

GLFWwindow* window;

// Camera intrinsic parameters
static cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1.66917519e+03, 0.0, 9.58599981e+02,
                                                    0.0, 1.66758795e+03, 6.37204097e+02,
                                                    0.0, 0.0, 1.0);
static cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << -5.84632887e-02, 7.84458638e-01, -9.94221022e-04, -1.60610409e-04, -3.35437898e+00);
static const float markerLength = 0.05;
// Our vertices. Tree consecutive floats give a 3D vertex; Three consecutive vertices give a triangle.
// A cube has 6 faces with 2 triangles each, so this makes 6*2=12 triangles, and 12*3 vertices
static const GLfloat g_vertex_buffer_data[] = { 
	-1.0f,-1.0f,-1.0f,
	-1.0f,-1.0f, 1.0f,
	-1.0f, 1.0f, 1.0f,
	1.0f, 1.0f,-1.0f,
	-1.0f,-1.0f,-1.0f,
	-1.0f, 1.0f,-1.0f,
	1.0f,-1.0f, 1.0f,
	-1.0f,-1.0f,-1.0f,
	1.0f,-1.0f,-1.0f,
	1.0f, 1.0f,-1.0f,
	1.0f,-1.0f,-1.0f,
	-1.0f,-1.0f,-1.0f,
	-1.0f,-1.0f,-1.0f,
	-1.0f, 1.0f, 1.0f,
	-1.0f, 1.0f,-1.0f,
	1.0f,-1.0f, 1.0f,
	-1.0f,-1.0f, 1.0f,
	-1.0f,-1.0f,-1.0f,
	-1.0f, 1.0f, 1.0f,
	-1.0f,-1.0f, 1.0f,
	1.0f,-1.0f, 1.0f,
	1.0f, 1.0f, 1.0f,
	1.0f,-1.0f,-1.0f,
	1.0f, 1.0f,-1.0f,
	1.0f,-1.0f,-1.0f,
	1.0f, 1.0f, 1.0f,
	1.0f,-1.0f, 1.0f,
	1.0f, 1.0f, 1.0f,
	1.0f, 1.0f,-1.0f,
	-1.0f, 1.0f,-1.0f,
	1.0f, 1.0f, 1.0f,
	-1.0f, 1.0f,-1.0f,
	-1.0f, 1.0f, 1.0f,
	1.0f, 1.0f, 1.0f,
	-1.0f, 1.0f, 1.0f,
	1.0f,-1.0f, 1.0f
};
// One color for each vertex. They were generated randomly.
static const GLfloat g_color_buffer_data[] = { 
		0.583f,  0.771f,  0.014f,
		0.609f,  0.115f,  0.436f,
		0.327f,  0.483f,  0.844f,
		0.822f,  0.569f,  0.201f,
		0.435f,  0.602f,  0.223f,
		0.310f,  0.747f,  0.185f,
		0.597f,  0.770f,  0.761f,
		0.559f,  0.436f,  0.730f,
		0.359f,  0.583f,  0.152f,
		0.483f,  0.596f,  0.789f,
		0.559f,  0.861f,  0.639f,
		0.195f,  0.548f,  0.859f,
		0.014f,  0.184f,  0.576f,
		0.771f,  0.328f,  0.970f,
		0.406f,  0.615f,  0.116f,
		0.676f,  0.977f,  0.133f,
		0.971f,  0.572f,  0.833f,
		0.140f,  0.616f,  0.489f,
		0.997f,  0.513f,  0.064f,
		0.945f,  0.719f,  0.592f,
		0.543f,  0.021f,  0.978f,
		0.279f,  0.317f,  0.505f,
		0.167f,  0.620f,  0.077f,
		0.347f,  0.857f,  0.137f,
		0.055f,  0.953f,  0.042f,
		0.714f,  0.505f,  0.345f,
		0.783f,  0.290f,  0.734f,
		0.722f,  0.645f,  0.174f,
		0.302f,  0.455f,  0.848f,
		0.225f,  0.587f,  0.040f,
		0.517f,  0.713f,  0.338f,
		0.053f,  0.959f,  0.120f,
		0.393f,  0.621f,  0.362f,
		0.673f,  0.211f,  0.457f,
		0.820f,  0.883f,  0.371f,
		0.982f,  0.099f,  0.879f
	};


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
    if (!initializeGLFWAndGLEW(&window, width, height)) {
		return -1;
	}
	setupOpenGLRendering();

    GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

    // Create and compile our GLSL program from the shaders
	GLuint programID = LoadShaders( 
		"../TransformVertexShader.vertexshader", 
		"../ColorFragmentShader.fragmentshader" 
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
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1920, 1080, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
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
                solvePnP(objPoints, corners.at(i), cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
				cv::drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
				MVP = getMVPMatrix(rvecs[i], tvecs[i], cameraMatrix, 1920, 1080);

				glBindFramebuffer(GL_FRAMEBUFFER, fbo);
				render(programID, MatrixID, MVP, vertexbuffer, colorbuffer);
				glBindFramebuffer(GL_FRAMEBUFFER, 0);

				// Read the pixels from the texture into an OpenCV Mat
				cv::Mat renderedImage(1080, 1920, CV_8UC3);
				glBindTexture(GL_TEXTURE_2D, renderedTexture);
				glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, renderedImage.data);
				cv::flip(renderedImage, renderedImage, 0); // Flip the image vertically

				// Overlay the OpenGL rendered image onto the captured frame
				cv::addWeighted(imageCopy, 1.0, renderedImage, 0.8, 0, imageCopy);
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
		glVertexAttribPointer(
			0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		// 2nd attribute buffer : colors
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
		glVertexAttribPointer(
			1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			3,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);

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
	glm::mat4 Projection = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.0f);
	// Camera matrix
	glm::mat4 View       = glm::lookAt(
								glm::vec3(4,3,-3), // Camera is at (4,3,-3), in World Space
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
    // Convert OpenCV rotation matrix to OpenGL model matrix
    glm::mat4 model = glm::mat4(1.0f);
    model[0][0] = static_cast<float>(R.at<double>(0, 0));
    model[0][1] = static_cast<float>(-R.at<double>(1, 0)); // Invert the y-axis
    model[0][2] = static_cast<float>(-R.at<double>(2, 0)); // Invert the z-axis
    model[0][3] = 0.0f;

    model[1][0] = static_cast<float>(R.at<double>(0, 1));
    model[1][1] = static_cast<float>(-R.at<double>(1, 1)); // Invert the y-axis
    model[1][2] = static_cast<float>(-R.at<double>(2, 1)); // Invert the z-axis
    model[1][3] = 0.0f;

    model[2][0] = static_cast<float>(R.at<double>(0, 2));
    model[2][1] = static_cast<float>(-R.at<double>(1, 2)); // Invert the y-axis
    model[2][2] = static_cast<float>(-R.at<double>(2, 2)); // Invert the z-axis
    model[2][3] = 0.0f;

    model[3][0] = static_cast<float>(tvec[0]*10.0);
    model[3][1] = static_cast<float>(-tvec[1]*10.0); // Invert the y-axis
    model[3][2] = static_cast<float>(-tvec[2]*10.0); // Invert the z-axis
    model[3][3] = 1.0f;

    // Create projection matrix based on the camera intrinsics
    float fx = static_cast<float>(cameraMatrix.at<double>(0, 0));
    float fy = static_cast<float>(cameraMatrix.at<double>(1, 1));
    float cx = static_cast<float>(cameraMatrix.at<double>(0, 2));
    float cy = static_cast<float>(cameraMatrix.at<double>(1, 2));

    glm::mat4 projection = glm::mat4(1.0f); // Identity matrix
    projection[0][0] = 2.0f * fx / width;
    projection[1][1] = 2.0f * fy / height;
    projection[2][2] = -(farPlane + nearPlane) / (farPlane - nearPlane);
    projection[2][3] = -1.0f;
    projection[3][2] = -2.0f * farPlane * nearPlane / (farPlane - nearPlane);
    projection[0][2] = 1.0f - 2.0f * cx / width;  // Adjusted for OpenGL's coordinate system
    projection[1][2] = 1.0f - 2.0f * cy / height; // Adjusted for OpenGL's coordinate system


    // Combine model and projection matrices
    glm::mat4 MVP = projection * model;

    return MVP;
}

void getWidthandHeight(cv::Mat image, int* width, int* height) {
    if (width != nullptr && height != nullptr) {
        *width = image.cols;
        *height = image.rows;
    }
}

