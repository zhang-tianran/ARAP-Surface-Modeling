#pragma once

#include "graphics/shape.h"
#include "Eigen/StdList"
#include "Eigen/StdVector"
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <set>

typedef Eigen::SparseMatrix<float> SpMat;
typedef Eigen::Triplet<float> T;

class Shader;

class ARAP
{
private:
    Shape m_shape;

    int n; // vert size
    int fixed; // anchor size

    std::unordered_set<int> anchors;
    std::vector<Eigen::Vector3f> verts;
    std::map<int, std::unordered_set<int>> neighbors;

    std::map<std::pair<int, int>, float> weights;
    std::vector<Eigen::Matrix3f> rotation;
    std::vector<Eigen::Vector3f> vert_primes;
    std::vector<int> idx_map;

    void initNeighbor();
    float getWeight(int i, int j);
    std::unordered_set<int> getSharedVert(std::unordered_set<int> a, std::unordered_set<int> b);

    Eigen::Matrix3f getS(int i);
    void calcR();
    Eigen::MatrixXf calcB();
    SpMat calcL();

    void constrain();
    void updatePrime(Eigen::MatrixXf prime);

    Eigen::SimplicialLLT<SpMat> llt;
    Eigen::JacobiSVD<Eigen::MatrixXf> svd;

public:
    ARAP();

    void init(Eigen::Vector3f &min, Eigen::Vector3f &max);
    void move(int vertex, Eigen::Vector3f pos);
    void update();


    // ================== Students, If You Choose To Modify The Code Below, It's On You

    int getClosestVertex(Eigen::Vector3f start, Eigen::Vector3f ray, float threshold)
    {
        return m_shape.getClosestVertex(start, ray, threshold);
    }

    void draw(Shader *shader, GLenum mode)
    {
        m_shape.draw(shader, mode);
    }

    SelectMode select(Shader *shader, int vertex)
    {
        return m_shape.select(shader, vertex);
    }

    bool selectWithSpecifiedMode(Shader *shader, int vertex, SelectMode mode)
    {
        return m_shape.selectWithSpecifiedMode(shader, vertex, mode);
    }

    bool getAnchorPos(int lastSelected, Eigen::Vector3f& pos, Eigen::Vector3f ray, Eigen::Vector3f start)
    {
        return m_shape.getAnchorPos(lastSelected, pos, ray, start);
    }
};
