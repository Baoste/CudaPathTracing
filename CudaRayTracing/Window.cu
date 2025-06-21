
#include <iostream>
#include "Window.cuh"

Window::Window(int w, int h, Camera* _camera) : width(w), height(h), tex(0)
{
    camera = _camera;
    sampleCount = 1;
    window = nullptr;
}

Window::~Window()
{
    glDeleteTextures(1, &tex);
    glfwDestroyWindow(window);
    glfwTerminate();
}

GLuint Window::createShaderProgram()
{
    const char* vertex_shader_code = 
R"(#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
)";

    const char* fragment_shader_code = 
R"(#version 330 core
in vec2 TexCoord;
uniform sampler2D tex;
out vec4 FragColor;
void main()
{
    FragColor = texture(tex, TexCoord);
}
)";

    GLint success;
    char info_log[512];

    // 顶点着色器
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertex_shader_code, nullptr);
    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, nullptr, info_log);
        std::cerr << "ERROR::VERTEX_SHADER::COMPILATION_FAILED\n" << info_log << std::endl;
    }

    // 片元着色器
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragment_shader_code, nullptr);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, info_log);
        std::cerr << "ERROR::FRAGMENT_SHADER::COMPILATION_FAILED\n" << info_log << std::endl;
    }

    // 链接 shader program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, info_log);
        std::cerr << "ERROR::SHADER_PROGRAM::LINK_FAILED\n" << info_log << std::endl;
    }

    // 删除临时 shader
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

void Window::setupTexturedQuad()
{
    float vertices[] = {
        // positions         // texCoords
         1.0f,  1.0f, 0.0f,   1.0f, 1.0f, // top right
         1.0f, -1.0f, 0.0f,   1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 0.0f,   0.0f, 1.0f  // top left
    };
    unsigned int indices[] = {
        0, 1, 3,
        1, 2, 3
    };

    shaderProgram = createShaderProgram();
    glUseProgram(shaderProgram);
    // generate vertex array object
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    // generate vertex buffer object
    glGenBuffers(1, &VBO);
    // generate element buffer object
    glGenBuffers(1, &EBO);
    // bind the vertex array object
    glBindVertexArray(VAO);
    // copy vertices array into vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // copy indices array into element buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    // configure vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
        (void*) nullptr);
    glEnableVertexAttribArray(0); // pos
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
        (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1); // texture coords

    // texture
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // 注册 OpenGL 缓冲区到 CUDA 图形资源
    glGenBuffers(1, &bufferObj);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferObj);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);

    size_t size;
    // 映射资源
    cudaGraphicsMapResources(1, &resource, NULL);
    cudaGraphicsResourceGetMappedPointer((void**)&devicePtr, &size, resource);
    // we should immediately unmap the resource, according to the RULES
    cudaGraphicsUnmapResources(1, &resource, NULL);
}

bool Window::Init()
{
    if (!glfwInit())
        return false;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

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

    setupTexturedQuad();
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);

    // 初始化 ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark(); // 设置暗色主题

    // 初始化平台/渲染器绑定
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    return true;
}

bool Window::Close()
{
    return glfwWindowShouldClose(window);
}

void Window::Update()
{
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT);

    // 上传纹理
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferObj);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    glUseProgram(0);
    glBindVertexArray(0);     // 清除 VAO
    glBindTexture(GL_TEXTURE_2D, 0); // 清除纹理

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

    // 渲染
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
        theta += moveSpeed * deltaTime;
        moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        theta -= moveSpeed * deltaTime;
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

