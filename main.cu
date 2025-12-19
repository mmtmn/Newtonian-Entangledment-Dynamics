// nvcc main.cu -o main -lGL -lGLU -lGLEW -lglfw -lglut
// ./main.cu

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glu.h>
#include <GL/glut.h>   // <<< ADD

#include <cuda_runtime.h>

#include <vector>
#include <cmath>
#include <cstdio>

/* ================= CUDA ================= */

struct SphereGPU {
    float3 omega;
};

__device__ float3 crossf(float3 a, float3 b) {
    return make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

__global__ void enforce_mesh(SphereGPU* a, SphereGPU* b) {
    float softness = 0.05f;
    float damping  = 0.995f;
    float maxW     = 5.0f;

    float3 v1 = crossf(a->omega, make_float3( 1,0,0));
    float3 v2 = crossf(b->omega, make_float3(-1,0,0));

    float3 vrel = make_float3(
        v1.x - v2.x,
        v1.y - v2.y,
        v1.z - v2.z
    );

    a->omega.y -= vrel.y * softness;
    a->omega.z -= vrel.z * softness;
    b->omega.y += vrel.y * softness;
    b->omega.z += vrel.z * softness;

    a->omega.x *= damping;
    a->omega.y *= damping;
    a->omega.z *= damping;
    b->omega.x *= damping;
    b->omega.y *= damping;
    b->omega.z *= damping;

    a->omega.x = fminf(fmaxf(a->omega.x, -maxW), maxW);
    a->omega.y = fminf(fmaxf(a->omega.y, -maxW), maxW);
    a->omega.z = fminf(fmaxf(a->omega.z, -maxW), maxW);
    b->omega.x = fminf(fmaxf(b->omega.x, -maxW), maxW);
    b->omega.y = fminf(fmaxf(b->omega.y, -maxW), maxW);
    b->omega.z = fminf(fmaxf(b->omega.z, -maxW), maxW);
}

/* ================= QUATERNIONS ================= */

struct Quat {
    float w,x,y,z;
};

Quat quat_mul(Quat a, Quat b) {
    return {
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
}

Quat quat_norm(Quat q) {
    float n = sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    return { q.w/n, q.x/n, q.y/n, q.z/n };
}

void integrate_quat(Quat& q, float3 w, float dt) {
    Quat dq{0, w.x*dt, w.y*dt, w.z*dt};
    q = quat_norm(quat_mul(q, dq));
}

void quat_to_mat(const Quat& q, float M[16]) {
    float xx=q.x*q.x, yy=q.y*q.y, zz=q.z*q.z;
    float xy=q.x*q.y, xz=q.x*q.z, yz=q.y*q.z;
    float wx=q.w*q.x, wy=q.w*q.y, wz=q.w*q.z;

    M[0]=1-2*(yy+zz); M[4]=2*(xy-wz);   M[8]=2*(xz+wy);   M[12]=0;
    M[1]=2*(xy+wz);   M[5]=1-2*(xx+zz); M[9]=2*(yz-wx);   M[13]=0;
    M[2]=2*(xz-wy);   M[6]=2*(yz+wx);   M[10]=1-2*(xx+yy);M[14]=0;
    M[3]=0;           M[7]=0;           M[11]=0;          M[15]=1;
}

/* ================= GEOMETRY ================= */

struct Vert { float x,y,z,r,g,b; };

GLuint makeToothedSphere(int seg, int rings, int teeth, int& count) {
    std::vector<Vert> v;

    for(int i=0;i<rings;i++){
        float p0=M_PI*i/rings;
        float p1=M_PI*(i+1)/rings;

        for(int j=0;j<=seg;j++){
            float t=2*M_PI*j/seg;
            bool tooth = (j % (seg/teeth)) < (seg/(teeth*2));
            float c = tooth ? 0.5f : 0.85f;

            auto emit=[&](float p){
                v.push_back({
                    sin(p)*cos(t),
                    cos(p),
                    sin(p)*sin(t),
                    c,c,c
                });
            };

            emit(p0);
            emit(p1);
        }
    }

    count = v.size();
    GLuint vbo;
    glGenBuffers(1,&vbo);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glBufferData(GL_ARRAY_BUFFER,v.size()*sizeof(Vert),v.data(),GL_STATIC_DRAW);
    return vbo;
}

/* ================= TEXT ================= */

void drawText(float x, float y, const char* s) {
    glRasterPos2f(x,y);
    while(*s) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18,*s++);
}

/* ================= MAIN ================= */

int main(int argc, char** argv){
    glutInit(&argc, argv);  // <<< ADD

    glfwInit();
    GLFWwindow* w=glfwCreateWindow(1200,800,"Newtonian Entangledment Dynamics",0,0);
    glfwMakeContextCurrent(w);
    glewInit();

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.05f,0.06f,0.08f,1);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60,1200.0/800.0,0.1,100);
    glMatrixMode(GL_MODELVIEW);

    int vc;
    GLuint vbo=makeToothedSphere(96,64,16,vc);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(3,GL_FLOAT,sizeof(Vert),(void*)0);
    glColorPointer(3,GL_FLOAT,sizeof(Vert),(void*)(3*sizeof(float)));

    float x1=-2,x2=2;
    Quat q1{1,0,0,0}, q2{1,0,0,0};

    SphereGPU h1{{0.4f,0.9f,0.2f}};
    SphereGPU h2{{-0.3f,0.2f,0.8f}};
    SphereGPU *d1,*d2;
    cudaMalloc(&d1,sizeof(h1));
    cudaMalloc(&d2,sizeof(h2));
    cudaMemcpy(d1,&h1,sizeof(h1),cudaMemcpyHostToDevice);
    cudaMemcpy(d2,&h2,sizeof(h2),cudaMemcpyHostToDevice);

    char buf[64];

    while(!glfwWindowShouldClose(w)){
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();
        gluLookAt(0,3,10, 0,0,0, 0,1,0);

        if(x1 < -1.01f){ x1+=0.01f; x2-=0.01f; }
        else enforce_mesh<<<1,1>>>(d1,d2);

        cudaMemcpy(&h1,d1,sizeof(h1),cudaMemcpyDeviceToHost);
        cudaMemcpy(&h2,d2,sizeof(h2),cudaMemcpyDeviceToHost);

        integrate_quat(q1,h1.omega,0.01f);
        integrate_quat(q2,h2.omega,0.01f);

        float M[16];

        glPushMatrix();
        glTranslatef(x1,0,0);
        quat_to_mat(q1,M);
        glMultMatrixf(M);
        glDrawArrays(GL_TRIANGLE_STRIP,0,vc);
        glPopMatrix();

        glPushMatrix();
        glTranslatef(x2,0,0);
        quat_to_mat(q2,M);
        glMultMatrixf(M);
        glDrawArrays(GL_TRIANGLE_STRIP,0,vc);
        glPopMatrix();

        /* ---------- HUD ---------- */
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        gluOrtho2D(0,1,0,1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glColor3f(0,1,0);
        snprintf(buf,64,"LEFT  %.2f  %s",
            fabs(h1.omega.z),
            h1.omega.z > 0 ? "⟲" : "⟳");
        drawText(0.05f,0.95f,buf);

        glColor3f(0.2f,0.4f,1.0f);
        snprintf(buf,64,"RIGHT %.2f  %s",
            fabs(h2.omega.z),
            h2.omega.z > 0 ? "⟲" : "⟳");
        drawText(0.70f,0.95f,buf);

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        /* -------------------------- */

        glfwSwapBuffers(w);
        glfwPollEvents();
    }
}
