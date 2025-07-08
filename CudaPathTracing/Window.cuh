#pragma once

#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "./imgui/imgui.h"
#include "./imgui/imgui_impl_glfw.h"
#include "./imgui/imgui_impl_opengl3.h"
#include <cuda_gl_interop.h>

#include "Camera.cuh"
#include "Scene.cuh"
#include "Render.cuh"


class Window
{
public:
    int width;
    int height;

    GLFWwindow* window;
    GLuint tex;

    bool spacePressed = false;
    Camera* camera;
    static Scene* scene;

    double deltaTime;

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
    // 各向异性
    static float alphaX;
    static float alphaY;
    // spp
    static int selectSampleCount;
    // 玻璃材质
    static bool glass;
    // 渲染方式
    const char* renderType[4];
    static int currentType;
    // fps曲线
    static const int fpsCurveSize = 30;
    float fpsCurve[fpsCurveSize] = { 0.0f };
    int fpsIndex = 0;
    // sigma for gaussian
    static float sigmaG;
    static float sigmaR;
    static float sigmaN;
    static float sigmaD;

public:
    Window(int w, int h, Camera* _camera, Scene* scene);
    ~Window();
    bool Init();
    bool Close();
    void Update();
    bool PollInput();

private:
    GLuint createShaderProgram();
    void setupTexturedQuad();
};

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);