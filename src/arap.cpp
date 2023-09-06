#include "arap.h"
#include "graphics/meshloader.h"

#include <iostream>
#include <set>
#include <map>
#include <vector>

using namespace std;
using namespace Eigen;

ARAP::ARAP() {}

void ARAP::init(Eigen::Vector3f &coeffMin, Eigen::Vector3f &coeffMax)
{
    vector<Vector3f> vertices;
    vector<Vector3i> triangles;

    // If this doesn't work for you, remember to change your working directory
    if (MeshLoader::loadTriMesh("meshes/peter.obj", vertices, triangles)) {
        m_shape.init(vertices, triangles);
    }

    // Students, please don't touch this code: get min and max for viewport stuff
    MatrixX3f all_vertices = MatrixX3f(vertices.size(), 3);
    int i = 0;
    for (unsigned long i = 0; i < vertices.size(); ++i) {
        all_vertices.row(i) = vertices[i];
    }
    coeffMin = all_vertices.colwise().minCoeff();
    coeffMax = all_vertices.colwise().maxCoeff();

    n = vertices.size();

    initNeighbor();

    // cash
    verts = vertices;
    anchors = m_shape.getAnchors();
    n = verts.size();
    fixed = anchors.size();

    // init fixed length variables
    vert_primes = {};
    for (Vector3f v: vertices) {
        vert_primes.push_back(v);
    }

    for (int i = 0; i < n; i++) {
        idx_map.push_back(-1);
        rotation.push_back(Matrix3f::Zero());
    }
    constrain();
}

// build neighbor map
void ARAP::initNeighbor(){
    const std::vector<Vector3i>& faces = m_shape.getFaces();
    for (int i = 0; i < faces.size(); i++) {
        Vector3i f = faces[i];
        int v0 = f[0];
        int v1 = f[1];
        int v2 = f[2];
        // neighbor
        neighbors[v0].insert(v1);
        neighbors[v0].insert(v2);
        neighbors[v1].insert(v0);
        neighbors[v1].insert(v2);
        neighbors[v2].insert(v0);
        neighbors[v2].insert(v1);
    }
}

std::unordered_set<int> ARAP::getSharedVert(std::unordered_set<int> a, std::unordered_set<int> b){
    std::unordered_set<int> c;
    for (auto i = a.begin(); i != a.end(); i++) {
        if (b.find(*i) != b.end()){
            c.insert(*i);
        }
    }
    return c;
}

float ARAP::getWeight(int i, int j) {
    std::pair<int, int> pair = i < j ? std::make_pair(i, j) : std::make_pair(j, i);
    return weights[pair];
}

// L (9)
SpMat ARAP::calcL(){
    std::vector<T> tripletL;
    // init weights
    for (int i = 0; i < n; i++){
        const std::unordered_set<int> nbr_i = neighbors[i];
        int idx_i = idx_map[i];
        float sum = 0;
        // calc w = 0.5 * (cot(a) + cot(b))
        for (int j: nbr_i) {
            if (i < j) {
                float w = 0.f;
                const std::unordered_set<int> nbr_j = neighbors[j];
                const std::unordered_set<int> shared_vert = getSharedVert(nbr_i, nbr_j);
                for (auto k = shared_vert.begin(); k != shared_vert.end(); k++) {
                    // get two edges
                    Vector3f e1 = verts[i] - verts[*k];
                    Vector3f e2 = verts[j] - verts[*k];
                    // calc cot
                    float cot = abs(e1.dot(e2) / e1.cross(e2).norm());
                    w += cot;
                }
                if (shared_vert.size() > 0) {
                    w /= shared_vert.size();
                }
                weights[std::make_pair(i, j)] = w;
            }
            // L matrix weight
            float w = getWeight(i, j);
            int idx_j = idx_map[j];
            if (idx_i != -1 && idx_j != -1) {
                tripletL.push_back(T(idx_i, idx_j, -w));
            }
            sum += w;
        }
        // diag weight
        if (idx_i != -1) {
            tripletL.push_back(T(idx_i, idx_i, sum));
        }
    }

    // build sparse L
    SpMat L(n - fixed, n - fixed);
    L.setFromTriplets(tripletL.begin(), tripletL.end());
    return L;
}

// b (9)
MatrixXf ARAP::calcB(){
    calcR();
    MatrixXf b = MatrixXf::Zero(n - fixed, 3);
    for (int i = 0; i < n; i++) {
        int idx_i = idx_map[i];
        const std::unordered_set<int> nbr = neighbors[i];
        if (idx_i != -1) {
            Vector3f bi(0, 0, 0);
            for (int j: nbr) {
                float w = 0.5f * getWeight(i, j);
                Matrix3f r = rotation[i] + rotation[j];
                Vector3f p = verts[i] - verts[j];
                bi += w * r * p;
            }
            b.row(idx_i) += bi;
        } else {
            for (int j: nbr) {
                int idx_j = idx_map[j];
                if (idx_j != -1) {
                    b.row(idx_j) += getWeight(i, j) * vert_primes[i];
                }
            }
        }
    }
    return b;
}

// covariance s = p * D * p'T (5)
Matrix3f ARAP::getS(int i){
    const std::unordered_set<int> nbr = neighbors[i];
    MatrixXf p = MatrixXf::Zero(3, nbr.size());
    MatrixXf D = MatrixXf::Zero(nbr.size(), nbr.size());
    MatrixXf p_prime = MatrixXf::Zero(3, nbr.size());
    auto itr = nbr.begin();
    for (int j = 0; j < nbr.size(); j++) {
        p.col(j) = verts[i] - verts[*itr];
        D(j, j) = getWeight(i, *itr);
        p_prime.col(j) = vert_primes[i] - vert_primes[*itr];
        itr++;
    }
    return p * D * p_prime.transpose();
}

// rotation R = U * VT (6)
void ARAP::calcR(){
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        Matrix3f S = getS(i);
        svd.compute(S, ComputeThinU | ComputeThinV);
        Matrix3f U = svd.matrixU();
        Matrix3f V = svd.matrixV();
        Matrix3f R = V * U.transpose();
        if (R.determinant() < 0) {
            U(U.rows() - 1, U.cols() - 1) *= -1.f;
            R = V * U.transpose();
        }
        rotation[i] = R;
    }
}

void ARAP::constrain(){
    int count = 0;
    for (int i = 0; i < n; i++){
        if (anchors.contains(i)) {
            idx_map[i] =  -1;
        } else {
            idx_map[i] = count;
            count += 1;
        }
    }
}

void ARAP::updatePrime(MatrixXf prime){
    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        if (idx_map[i] != -1) {
            vert_primes[i] = prime.row(idx_map[i]);
        }
    }
}

void ARAP::update(){
    // cash
    verts = m_shape.getVertices();
    anchors = m_shape.getAnchors();
    fixed = anchors.size();
    constrain();
    // update L
    SpMat L = calcL();
    llt.compute(L);
}

// Move an anchored vertex, defined by its index, to targetPosition
void ARAP::move(int vertex, Vector3f targetPosition)
{
    // Here are some helpful controls for the application
    //
    // - You start in first-person camera mode
    //   - WASD to move, left-click and drag to rotate
    //   - R and F to move vertically up and down
    //
    // - C to change to orbit camera mode
    //
    // - Right-click (and, optionally, drag) to anchor/unanchor points
    //   - Left-click an anchored point to move it around
    //
    // - Minus and equal keys (click repeatedly) to change the size of the vertices

    vert_primes[vertex] = targetPosition;

    for (int i = 0; i < 4; i++) {
        MatrixXf b = calcB();
        MatrixXf prime = llt.solve(b);
        updatePrime(prime);
    }

    m_shape.setVertices(vert_primes);
}
