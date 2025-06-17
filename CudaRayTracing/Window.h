#pragma once

#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "./imgui/imgui.h"
#include "./imgui/imgui_impl_glfw.h"
#include "./imgui/imgui_impl_opengl3.h"

class Window
{
private:
    int width;
    int height;

    GLFWwindow* window;
    GLuint tex;

    bool spacePressed = false;

public:
    unsigned char* img;
    size_t cb_size;
    bool paused = false;
    int sampleCount;

public:
    double roughness = 1.0f;
    double metallic = 0.0f;

public:
    Window(int w, int h);
    ~Window();
    bool Init();
    bool Close();
    void Update();
    void PollInput();
};

