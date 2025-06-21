#pragma once

#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "./imgui/imgui.h"
#include "./imgui/imgui_impl_glfw.h"
#include "./imgui/imgui_impl_opengl3.h"
#include <cuda_gl_interop.h>

#include "Camera.cuh"

class Window
{
public:
    int width;
    int height;

    GLFWwindow* window;
    GLuint tex;

    bool spacePressed = false;
    Camera* camera;

public:
    bool paused = false;
    int sampleCount;
    uchar4* devicePtr;

public:
    GLuint bufferObj;
    cudaGraphicsResource* resource;
    unsigned int VBO, VAO, EBO;
    GLuint shaderProgram;

public:
    double roughness = 1.0f;
    double metallic = 0.0f;

public:
    Window(int w, int h, Camera* _camera);
    ~Window();
    bool Init();
    bool Close();
    void Update();
    bool PollInput();

private:
    GLuint createShaderProgram();
    void setupTexturedQuad();
};

