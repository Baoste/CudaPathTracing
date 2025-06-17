
#include "Window.h"
#include <iostream>

Window::Window(int w, int h) : width(w), height(h), tex(0)
{
    cb_size = 4 * w * h * sizeof(unsigned char);
    img = (unsigned char*)malloc(cb_size);

    sampleCount = 1;
    window = nullptr;
}

Window::~Window()
{
    glDeleteTextures(1, &tex);
    glfwDestroyWindow(window);
    glfwTerminate();
}

bool Window::Init()
{
    if (!glfwInit())
        return false;

    window = glfwCreateWindow(width, height, "CUDA + OpenGL", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);

    // 初始化 GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD 初始化失败\n";
        return false;
    }

    // 创建纹理
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 设置正交视口和2D quad
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width, 0, height, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);

    // 初始化 ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark(); // 设置暗色主题

    // 初始化平台/渲染器绑定
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330"); // <- 这里在 GLAD 初始化后调用

    return true;
}

bool Window::Close()
{
    return glfwWindowShouldClose(window);
}

void Window::Update()
{
    glClear(GL_COLOR_BUFFER_BIT);

    // 每帧开始
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // GUI 窗口
    ImGui::Begin("Config");
    float roughness_f = static_cast<float>(roughness);
    float metallic_f = static_cast<float>(metallic);
    if (ImGui::SliderFloat("Roughness", &roughness_f, 0.0f, 1.0f))
        roughness = static_cast<double>(roughness_f);
    if (ImGui::SliderFloat("Metallic", &metallic_f, 0.0f, 1.0f))
        metallic = static_cast<double>(metallic_f);
    
    ImGui::End();


    // 上传纹理
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);

    // 绘制
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(width, 0.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(width, height);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, height);
    glEnd();
    
    // 渲染
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
    glfwPollEvents();
}


void Window::PollInput()
{
    glfwPollEvents();

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        if (!spacePressed)
        {
            paused = !paused;
            spacePressed = true;
            sampleCount = paused ? 64 : 1;
            std::cout << (paused ? "Paused\n" : "Resumed\n");
        }
    }
    else
    {
        spacePressed = false;
    }
}

