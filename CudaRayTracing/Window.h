#pragma once

#include <GLFW/glfw3.h>
#include <vector>

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
    Window(int w, int h);
    ~Window();
    bool Init();
    bool Close();
    void Update();
    void PollInput();
};

