
#include <iostream>
#include "Window.cuh"

Window::Window(int w, int h, Camera* _camera) : width(w), height(h), tex(0)
{
    camera = _camera;
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

    // ��ʼ�� GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD ��ʼ��ʧ��\n";
        return false;
    }

    // ��������
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // ���������ӿں�2D quad
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width, 0, height, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);

    // ��ʼ�� ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark(); // ���ð�ɫ����

    // ��ʼ��ƽ̨/��Ⱦ����
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330"); // <- ������ GLAD ��ʼ�������

    return true;
}

bool Window::Close()
{
    return glfwWindowShouldClose(window);
}

void Window::Update()
{
    glClear(GL_COLOR_BUFFER_BIT);

    // ÿ֡��ʼ
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // GUI ����
    ImGui::Begin("Config");
    float roughness_f = static_cast<float>(roughness);
    float metallic_f = static_cast<float>(metallic);
    if (ImGui::SliderFloat("Roughness", &roughness_f, 0.0f, 1.0f))
        roughness = static_cast<double>(roughness_f);
    if (ImGui::SliderFloat("Metallic", &metallic_f, 0.0f, 1.0f))
        metallic = static_cast<double>(metallic_f);
    
    ImGui::End();


    // �ϴ�����
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);

    // ����
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(width, 0.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(width, height);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, height);
    glEnd();
    
    // ��Ⱦ
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
    glfwPollEvents();
}


bool Window::PollInput()
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
            return true;
        }
    }
    else
    {
        spacePressed = false;
    }

    bool moved = false;
    double moveSpeed = 1.0;
    double deltaTime = 0.1;
    double phi = 0.0;
    double theta = 0.0;
    double x = 0.0;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        theta -= moveSpeed * deltaTime;
        moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        theta += moveSpeed * deltaTime;
        moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        phi -= moveSpeed * deltaTime;
        moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        phi += moveSpeed * deltaTime;
        moved = true;
    }
    if (moved)
    {
        camera->move(phi, theta, x);
        return true;
    }
    return false;
}

