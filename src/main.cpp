// Standard Library headers
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// Third-party library headers
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

// Project-specific headers
#include <include/global.hpp>
#include <include/OpenGLUtils.hpp>
#include <common/shader.hpp>
#include <common/texture.hpp>
#include <common/controls.hpp>

GLFWwindow* window;

void getWidthandHeight(cv::Mat image, int* width, int* height);

// glm::mat4 getProjectionMatrix(const cv::Mat& cameraMatrix, int width, int height);
// glm:: mat4 getMVPMatri

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
    if (!initOpenGL(&window, width/2, height/2)) {
		return -1;
	}
	setOpenGLRendering(window);

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

    glm::mat4 MVP;

    GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	GLuint colorbuffer;
	glGenBuffers(1, &colorbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_color_buffer_data), g_color_buffer_data, GL_STATIC_DRAW);

	GLuint fbo, renderedTexture;
    setFrameBuffer(fbo, renderedTexture, width, height);


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

				renderScene(window, programID, MatrixID, MVP, vertexbuffer, colorbuffer);

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


glm::mat4 getMVPMatrix(const cv::Vec3d& rvec, const cv::Vec3d& tvec, const cv::Mat& cameraMatrix, int width, int height) {
    float nearPlane = 0.01f;
    float farPlane = 100.0f;

    // Convert rotation vector to rotation matrix
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // Scale matrix
    glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(4.0f));

    // Rotation matrix
    // Invert the y-axis and z-axis
    glm::mat4 rotation = glm::mat4(1.0f);
    rotation[0][0] = static_cast<float>(R.at<double>(0, 0));
    rotation[0][1] = static_cast<float>(-R.at<double>(1, 0));
    rotation[0][2] = static_cast<float>(-R.at<double>(2, 0));

    rotation[1][0] = static_cast<float>(R.at<double>(0, 1));
    rotation[1][1] = static_cast<float>(-R.at<double>(1, 1));
    rotation[1][2] = static_cast<float>(-R.at<double>(2, 1));

    rotation[2][0] = static_cast<float>(R.at<double>(0, 2));
    rotation[2][1] = static_cast<float>(-R.at<double>(1, 2));
    rotation[2][2] = static_cast<float>(-R.at<double>(2, 2));

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

    // Combine model and projection matrices
    glm::mat4 MVP = projection * model;

    return MVP;
}

void getWidthandHeight(cv::Mat image, int* width, int* height) {
    if (width != nullptr && height != nullptr) {
        *width = image.cols;
        *height = image.rows;
        std::cout << "width: " << *width << " height: " << *height << std::endl;
    }
}

